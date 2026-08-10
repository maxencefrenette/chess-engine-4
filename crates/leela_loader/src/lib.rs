use std::collections::HashSet;
use std::collections::VecDeque;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, TryRecvError, sync_channel};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use flate2::read::GzDecoder;
use half::f16;
use numpy::{PyArray, PyArrayMethods};
use pyo3::exceptions::{PyStopIteration, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

mod converter;
mod parquet_loader;

pub(crate) const POLICY_SIZE: usize = 1858;
pub(crate) const COMPACT_POLICY_SIZE: usize = 218;
pub(crate) const HISTORY_PLANE_COUNT: usize = 104;
pub(crate) const BOARD_SIZE: usize = 8;
pub(crate) const PACKED_PLANE_BYTES: usize = HISTORY_PLANE_COUNT * BOARD_SIZE;
const VALUE_TYPE_COUNT: usize = 6;
const VALUE_FIELDS: usize = 3;
const VALUE_COUNT: usize = VALUE_TYPE_COUNT * VALUE_FIELDS;
pub(crate) const RECORD_SIZE: usize = 8356;
const PREFETCH_RECV_TIMEOUT: Duration = Duration::from_secs(5);

const VERSION_OFFSET: usize = 0;
pub(crate) const POLICY_OFFSET: usize = 8;
pub(crate) const PLANES_OFFSET: usize = POLICY_OFFSET + POLICY_SIZE * 4;
pub(crate) const CASTLING_US_OOO_OFFSET: usize = PLANES_OFFSET + PACKED_PLANE_BYTES;
pub(crate) const CASTLING_US_OO_OFFSET: usize = CASTLING_US_OOO_OFFSET + 1;
pub(crate) const CASTLING_THEM_OOO_OFFSET: usize = CASTLING_US_OO_OFFSET + 1;
pub(crate) const CASTLING_THEM_OO_OFFSET: usize = CASTLING_THEM_OOO_OFFSET + 1;
pub(crate) const SIDE_TO_MOVE_OFFSET: usize = CASTLING_THEM_OO_OFFSET + 1;
pub(crate) const RULE50_OFFSET: usize = SIDE_TO_MOVE_OFFSET + 1;
pub(crate) const ROOT_Q_OFFSET: usize = 8280;
const BEST_Q_OFFSET: usize = 8284;
pub(crate) const ROOT_D_OFFSET: usize = 8288;
const BEST_D_OFFSET: usize = 8292;
pub(crate) const ROOT_M_OFFSET: usize = 8296;
const BEST_M_OFFSET: usize = 8300;
const PLIES_LEFT_OFFSET: usize = 8304;
const RESULT_Q_OFFSET: usize = 8308;
const RESULT_D_OFFSET: usize = 8312;
const PLAYED_Q_OFFSET: usize = 8316;
const PLAYED_D_OFFSET: usize = 8320;
const PLAYED_M_OFFSET: usize = 8324;
const ORIG_Q_OFFSET: usize = 8328;
const ORIG_D_OFFSET: usize = 8332;
const ORIG_M_OFFSET: usize = 8336;

struct PackedBatchIterator {
    paths: Vec<PathBuf>,
    path_index: usize,
    reader: Option<SimpleTarReader>,
    pending: VecDeque<ChunkCursor>,
    pending_records: usize,
    batch_size: usize,
}

impl PackedBatchIterator {
    fn next_batch_data(&mut self) -> PyResult<Option<PackedBatchData>> {
        self.fill_pending()?;
        if self.pending_records == 0 {
            return Ok(None);
        }
        if self.pending_records < self.batch_size {
            self.pending.clear();
            self.pending_records = 0;
            return Ok(None);
        }

        let records = self.batch_size.min(self.pending_records);
        let batch = self.build_batch_data(records)?;
        self.pending_records -= records;
        Ok(Some(batch))
    }

    fn fill_pending(&mut self) -> PyResult<()> {
        while self.pending_records < self.batch_size {
            let Some(payload) = self.next_payload()? else {
                break;
            };
            if payload.is_empty() {
                continue;
            }
            if payload.len() % RECORD_SIZE != 0 {
                return Err(PyValueError::new_err(format!(
                    "LCZero chunk has {} bytes, not a multiple of {}",
                    payload.len(),
                    RECORD_SIZE
                )));
            }
            validate_versions(&payload)?;
            let records = payload.len() / RECORD_SIZE;
            if records == 0 {
                break;
            }
            self.pending.push_back(ChunkCursor {
                bytes: payload,
                record_offset: 0,
                records,
            });
            self.pending_records += records;
        }
        Ok(())
    }

    fn next_payload(&mut self) -> PyResult<Option<Vec<u8>>> {
        loop {
            if self.reader.is_none() {
                if self.path_index >= self.paths.len() {
                    return Ok(None);
                }
                let path = self.paths[self.path_index].clone();
                self.path_index += 1;
                self.reader = Some(SimpleTarReader::open(path)?);
            }
            let reader = self.reader.as_mut().expect("reader was just opened");
            match reader.next_regular_payload()? {
                Some(payload) => return Ok(Some(payload)),
                None => self.reader = None,
            }
        }
    }

    fn build_batch_data(&mut self, records: usize) -> PyResult<PackedBatchData> {
        let mut packed_planes = vec![0_u8; records * PACKED_PLANE_BYTES];
        let mut scalars = vec![0.0_f32; records * 8];
        let mut policy_indices = vec![-1_i16; records * COMPACT_POLICY_SIZE];
        let mut policy_probs = vec![f16::ZERO; records * COMPACT_POLICY_SIZE];
        let mut value = vec![0.0_f32; records * VALUE_COUNT];

        let mut out_record = 0;
        while out_record < records {
            let front = self
                .pending
                .front_mut()
                .expect("pending record count is non-zero");
            let available = front.records - front.record_offset;
            let take = available.min(records - out_record);
            for index in 0..take {
                let record = front.record(index);
                copy_record(
                    record,
                    out_record + index,
                    &mut packed_planes,
                    &mut scalars,
                    &mut policy_indices,
                    &mut policy_probs,
                    &mut value,
                )?;
            }
            front.record_offset += take;
            out_record += take;
            if front.record_offset == front.records {
                self.pending.pop_front();
            }
        }

        Ok(PackedBatchData {
            records,
            packed_planes,
            scalars,
            policy_indices,
            policy_probs,
            value,
        })
    }
}

pub(crate) struct PackedBatchData {
    pub(crate) records: usize,
    pub(crate) packed_planes: Vec<u8>,
    pub(crate) scalars: Vec<f32>,
    pub(crate) policy_indices: Vec<i16>,
    pub(crate) policy_probs: Vec<f16>,
    pub(crate) value: Vec<f32>,
}

impl PackedBatchData {
    fn into_py_tuple<'py>(self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let packed_planes = PyArray::from_vec(py, self.packed_planes).reshape([
            self.records,
            HISTORY_PLANE_COUNT,
            BOARD_SIZE,
        ])?;
        let scalars = PyArray::from_vec(py, self.scalars).reshape([self.records, 8])?;
        let policy_indices = PyArray::from_vec(py, self.policy_indices)
            .reshape([self.records, COMPACT_POLICY_SIZE])?;
        let policy_probs = PyArray::from_vec(py, self.policy_probs)
            .reshape([self.records, COMPACT_POLICY_SIZE])?;
        let value = PyArray::from_vec(py, self.value).reshape([
            self.records,
            VALUE_TYPE_COUNT,
            VALUE_FIELDS,
        ])?;
        PyTuple::new(
            py,
            [
                packed_planes.into_any(),
                scalars.into_any(),
                policy_indices.into_any(),
                policy_probs.into_any(),
                value.into_any(),
            ],
        )
    }
}

#[pyclass(module = "chess_engine_4_native", unsendable)]
pub(crate) struct PrefetchedPackedBatchIterator {
    pub(crate) receivers: Vec<Receiver<PrefetchMessage>>,
    pub(crate) active_receivers: Vec<bool>,
    pub(crate) active_receiver_count: usize,
    pub(crate) next_receiver: usize,
    pub(crate) stop: Arc<AtomicBool>,
    pub(crate) handles: Vec<JoinHandle<()>>,
}

#[pymethods]
impl PrefetchedPackedBatchIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let deadline = Instant::now() + PREFETCH_RECV_TIMEOUT;
        loop {
            if self.active_receiver_count == 0 {
                return Err(PyStopIteration::new_err(()));
            }
            for offset in 0..self.receivers.len() {
                let index = (self.next_receiver + offset) % self.receivers.len();
                if !self.active_receivers[index] {
                    continue;
                }
                match self.receivers[index].try_recv() {
                    Ok(PrefetchMessage::Batch(batch)) => {
                        self.next_receiver = (index + 1) % self.receivers.len();
                        return batch.into_py_tuple(py);
                    }
                    Ok(PrefetchMessage::End) | Err(TryRecvError::Disconnected) => {
                        self.active_receivers[index] = false;
                        self.active_receiver_count -= 1;
                    }
                    Ok(PrefetchMessage::Error(message)) => {
                        self.active_receivers[index] = false;
                        self.active_receiver_count -= 1;
                        return Err(PyValueError::new_err(message));
                    }
                    Err(TryRecvError::Empty) => {}
                }
            }
            if Instant::now() >= deadline {
                return Err(PyValueError::new_err(
                    "timed out after 5s waiting for the Rust dataloader prefetch threads",
                ));
            }
            thread::sleep(Duration::from_millis(1));
        }
    }
}

impl Drop for PrefetchedPackedBatchIterator {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        self.receivers.clear();
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}

pub(crate) enum PrefetchMessage {
    Batch(PackedBatchData),
    End,
    Error(String),
}

struct ChunkCursor {
    bytes: Vec<u8>,
    record_offset: usize,
    records: usize,
}

impl ChunkCursor {
    fn record(&self, offset: usize) -> &[u8] {
        let start = (self.record_offset + offset) * RECORD_SIZE;
        &self.bytes[start..start + RECORD_SIZE]
    }
}

pub(crate) struct SimpleTarReader {
    file: File,
    path: PathBuf,
}

impl SimpleTarReader {
    pub(crate) fn open(path: PathBuf) -> PyResult<Self> {
        let file = File::open(&path).map_err(|error| {
            PyValueError::new_err(format!("failed to open {}: {error}", path.display()))
        })?;
        Ok(Self { file, path })
    }

    pub(crate) fn next_regular_payload(&mut self) -> PyResult<Option<Vec<u8>>> {
        Ok(self.next_regular_entry()?.map(|(_, payload)| payload))
    }

    fn next_regular_entry(&mut self) -> PyResult<Option<(String, Vec<u8>)>> {
        loop {
            let mut header = [0_u8; 512];
            if !read_exact_or_eof(&mut self.file, &mut header).map_err(|error| {
                PyValueError::new_err(format!("failed to read {}: {error}", self.path.display()))
            })? {
                return Ok(None);
            }
            if header.iter().all(|byte| *byte == 0) {
                return Ok(None);
            }

            let size = parse_tar_size(&header[124..136])?;
            let typeflag = header[156];
            let name = tar_name(&header);
            let padded_size = size.div_ceil(512) * 512;

            if typeflag == b'0' || typeflag == 0 {
                let mut payload = vec![0_u8; size];
                self.file.read_exact(&mut payload).map_err(|error| {
                    PyValueError::new_err(format!(
                        "failed to read {} member {name}: {error}",
                        self.path.display()
                    ))
                })?;
                if padded_size > size {
                    self.file
                        .seek(SeekFrom::Current((padded_size - size) as i64))
                        .map_err(|error| {
                            PyValueError::new_err(format!(
                                "failed to seek {} member padding: {error}",
                                self.path.display()
                            ))
                        })?;
                }
                if name.rsplit('/').next() == Some("LICENSE") {
                    continue;
                }
                return Ok(decompress_if_gzip(payload)?.map(|payload| (name, payload)));
            }

            self.file
                .seek(SeekFrom::Current(padded_size as i64))
                .map_err(|error| {
                    PyValueError::new_err(format!(
                        "failed to skip {} member {name}: {error}",
                        self.path.display()
                    ))
                })?;
        }
    }
}

#[pyfunction]
fn iter_prefetched_packed_batches(
    paths: Vec<PathBuf>,
    batch_size: usize,
    prefetch_per_thread: usize,
    threads: usize,
) -> PyResult<PrefetchedPackedBatchIterator> {
    validate_batch_size(batch_size)?;
    if prefetch_per_thread == 0 {
        return Err(PyValueError::new_err(
            "prefetch_per_thread must be positive",
        ));
    }
    if threads == 0 {
        return Err(PyValueError::new_err("threads must be positive"));
    }

    let paths = Arc::new(Mutex::new(VecDeque::from(paths)));
    let stop = Arc::new(AtomicBool::new(false));
    let mut receivers = Vec::with_capacity(threads);
    let mut handles = Vec::with_capacity(threads);
    for _ in 0..threads {
        let (sender, receiver) = sync_channel(prefetch_per_thread);
        receivers.push(receiver);
        let worker_paths = Arc::clone(&paths);
        let worker_stop = Arc::clone(&stop);
        handles.push(thread::spawn(move || {
            loop {
                if worker_stop.load(Ordering::Relaxed) {
                    break;
                }
                let Some(path) = next_path(&worker_paths) else {
                    let _ = sender.send(PrefetchMessage::End);
                    break;
                };
                let mut iterator = make_packed_batch_iterator(vec![path], batch_size);
                loop {
                    if worker_stop.load(Ordering::Relaxed) {
                        return;
                    }
                    let message = match iterator.next_batch_data() {
                        Ok(Some(batch)) => PrefetchMessage::Batch(batch),
                        Ok(None) => break,
                        Err(error) => PrefetchMessage::Error(error.to_string()),
                    };
                    let done = matches!(message, PrefetchMessage::Error(_));
                    if sender.send(message).is_err() || done {
                        return;
                    }
                }
            }
        }));
    }

    let active_receiver_count = receivers.len();
    Ok(PrefetchedPackedBatchIterator {
        receivers,
        active_receivers: vec![true; active_receiver_count],
        active_receiver_count,
        next_receiver: 0,
        stop,
        handles,
    })
}

#[pyfunction]
fn iter_prefetched_parquet_batches(
    paths: Vec<PathBuf>,
    batch_size: usize,
    prefetch_per_thread: usize,
    threads: usize,
    sampling_rate: f64,
) -> PyResult<PrefetchedPackedBatchIterator> {
    validate_batch_size(batch_size)?;
    parquet_loader::iter_prefetched_parquet_batches(
        paths,
        batch_size,
        prefetch_per_thread,
        threads,
        sampling_rate,
    )
}

#[pyfunction]
fn convert_lc0_tar_to_parquet(input: PathBuf, output: PathBuf) -> PyResult<(usize, u64, u64)> {
    converter::convert_lc0_tar_to_parquet(input, output)
}

#[pyfunction]
fn parquet_row_counts(paths: Vec<PathBuf>) -> PyResult<Vec<(String, usize)>> {
    parquet_loader::parquet_row_counts(paths)
}

#[pyfunction]
fn inspect_lc0_tars(paths: Vec<PathBuf>) -> PyResult<(Vec<(String, usize, usize)>, usize)> {
    let mut game_names = HashSet::new();
    let mut duplicate_games = 0;
    let mut results = Vec::with_capacity(paths.len());
    for path in paths {
        let mut reader = SimpleTarReader::open(path.clone())?;
        let mut games = 0;
        let mut rows = 0;
        while let Some((name, payload)) = reader.next_regular_entry()? {
            if payload.len() % RECORD_SIZE != 0 {
                return Err(PyValueError::new_err(format!(
                    "LCZero chunk has {} bytes, not a multiple of {RECORD_SIZE}",
                    payload.len()
                )));
            }
            validate_versions(&payload)?;
            games += 1;
            rows += payload.len() / RECORD_SIZE;
            let game_id = name.rsplit('/').next().unwrap_or(&name).to_owned();
            if !game_names.insert(game_id) {
                duplicate_games += 1;
            }
        }
        results.push((path.to_string_lossy().into_owned(), games, rows));
    }
    Ok((results, duplicate_games))
}

fn next_path(paths: &Arc<Mutex<VecDeque<PathBuf>>>) -> Option<PathBuf> {
    paths.lock().expect("paths mutex poisoned").pop_front()
}

fn validate_batch_size(batch_size: usize) -> PyResult<()> {
    if batch_size == 0 {
        return Err(PyValueError::new_err("batch_size must be positive"));
    }
    Ok(())
}

fn make_packed_batch_iterator(paths: Vec<PathBuf>, batch_size: usize) -> PackedBatchIterator {
    PackedBatchIterator {
        paths,
        path_index: 0,
        reader: None,
        pending: VecDeque::new(),
        pending_records: 0,
        batch_size,
    }
}

#[pymodule]
fn chess_engine_4_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(iter_prefetched_packed_batches, module)?)?;
    module.add_function(wrap_pyfunction!(iter_prefetched_parquet_batches, module)?)?;
    module.add_function(wrap_pyfunction!(convert_lc0_tar_to_parquet, module)?)?;
    module.add_function(wrap_pyfunction!(parquet_row_counts, module)?)?;
    module.add_function(wrap_pyfunction!(inspect_lc0_tars, module)?)?;
    module.add("POLICY_SIZE", POLICY_SIZE)?;
    module.add("COMPACT_POLICY_SIZE", COMPACT_POLICY_SIZE)?;
    module.add("HISTORY_PLANE_COUNT", HISTORY_PLANE_COUNT)?;
    module.add("BOARD_SIZE", BOARD_SIZE)?;
    module.add("RECORD_SIZE", RECORD_SIZE)?;
    Ok(())
}

fn read_exact_or_eof(file: &mut File, buffer: &mut [u8]) -> std::io::Result<bool> {
    let mut read = 0;
    while read < buffer.len() {
        let count = file.read(&mut buffer[read..])?;
        if count == 0 {
            if read == 0 {
                return Ok(false);
            }
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "partial tar header",
            ));
        }
        read += count;
    }
    Ok(true)
}

fn parse_tar_size(bytes: &[u8]) -> PyResult<usize> {
    let nul_pos = bytes
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(bytes.len());
    let text = std::str::from_utf8(&bytes[..nul_pos])
        .map_err(|error| PyValueError::new_err(format!("invalid tar size bytes: {error}")))?
        .trim();
    if text.is_empty() {
        return Ok(0);
    }
    usize::from_str_radix(text, 8)
        .map_err(|error| PyValueError::new_err(format!("invalid tar size {text:?}: {error}")))
}

fn tar_name(header: &[u8; 512]) -> String {
    let name_end = header[..100]
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(100);
    String::from_utf8_lossy(&header[..name_end]).to_string()
}

fn decompress_if_gzip(payload: Vec<u8>) -> PyResult<Option<Vec<u8>>> {
    if payload.len() >= 2 && payload[0] == 0x1f && payload[1] == 0x8b {
        let mut decoder = GzDecoder::new(payload.as_slice());
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed).map_err(|error| {
            PyValueError::new_err(format!("gzip decompression failed: {error}"))
        })?;
        return Ok(Some(decompressed));
    }
    Ok(Some(payload))
}

fn validate_versions(payload: &[u8]) -> PyResult<()> {
    for record in payload.chunks_exact(RECORD_SIZE) {
        let version = read_u32(record, VERSION_OFFSET);
        if version != 6 {
            return Err(PyValueError::new_err(format!(
                "unsupported LCZero record version {version}"
            )));
        }
    }
    Ok(())
}

fn copy_record(
    record: &[u8],
    out_record: usize,
    packed_planes: &mut [u8],
    scalars: &mut [f32],
    policy_indices: &mut [i16],
    policy_probs: &mut [f16],
    value: &mut [f32],
) -> PyResult<()> {
    let planes_start = out_record * PACKED_PLANE_BYTES;
    packed_planes[planes_start..planes_start + PACKED_PLANE_BYTES]
        .copy_from_slice(&record[PLANES_OFFSET..PLANES_OFFSET + PACKED_PLANE_BYTES]);

    let scalars_start = out_record * 8;
    scalars[scalars_start] = record[CASTLING_US_OOO_OFFSET] as f32;
    scalars[scalars_start + 1] = record[CASTLING_US_OO_OFFSET] as f32;
    scalars[scalars_start + 2] = record[CASTLING_THEM_OOO_OFFSET] as f32;
    scalars[scalars_start + 3] = record[CASTLING_THEM_OO_OFFSET] as f32;
    scalars[scalars_start + 4] = record[SIDE_TO_MOVE_OFFSET] as f32;
    scalars[scalars_start + 5] = record[RULE50_OFFSET] as f32;
    scalars[scalars_start + 6] = 0.0;
    scalars[scalars_start + 7] = 1.0;

    let policy_start = out_record * COMPACT_POLICY_SIZE;
    let mut legal_count = 0;
    for policy_index in 0..POLICY_SIZE {
        let probability = read_f32(record, POLICY_OFFSET + policy_index * 4);
        if probability >= 0.0 {
            if legal_count == COMPACT_POLICY_SIZE {
                return Err(PyValueError::new_err(format!(
                    "record has more than {COMPACT_POLICY_SIZE} legal policy moves"
                )));
            }
            policy_indices[policy_start + legal_count] = policy_index as i16;
            policy_probs[policy_start + legal_count] = f16::from_f32(probability);
            legal_count += 1;
        }
    }

    let value_start = out_record * VALUE_COUNT;
    value[value_start] = read_f32(record, RESULT_Q_OFFSET);
    value[value_start + 1] = read_f32(record, RESULT_D_OFFSET);
    value[value_start + 2] = read_f32(record, PLIES_LEFT_OFFSET);
    value[value_start + 3] = read_f32(record, BEST_Q_OFFSET);
    value[value_start + 4] = read_f32(record, BEST_D_OFFSET);
    value[value_start + 5] = read_f32(record, BEST_M_OFFSET);
    value[value_start + 6] = read_f32(record, PLAYED_Q_OFFSET);
    value[value_start + 7] = read_f32(record, PLAYED_D_OFFSET);
    value[value_start + 8] = read_f32(record, PLAYED_M_OFFSET);
    value[value_start + 9] = read_f32(record, ORIG_Q_OFFSET);
    value[value_start + 10] = read_f32(record, ORIG_D_OFFSET);
    value[value_start + 11] = read_f32(record, ORIG_M_OFFSET);
    value[value_start + 12] = read_f32(record, ROOT_Q_OFFSET);
    value[value_start + 13] = read_f32(record, ROOT_D_OFFSET);
    value[value_start + 14] = read_f32(record, ROOT_M_OFFSET);
    value[value_start + 15] = 0.0;
    value[value_start + 16] = 0.0;
    value[value_start + 17] = f32::NAN;
    Ok(())
}

pub(crate) fn read_u32(record: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(
        record[offset..offset + 4]
            .try_into()
            .expect("u32 offset is valid"),
    )
}

pub(crate) fn read_f32(record: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes(
        record[offset..offset + 4]
            .try_into()
            .expect("f32 offset is valid"),
    )
}
