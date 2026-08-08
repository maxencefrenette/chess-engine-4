use std::collections::VecDeque;
use std::fs::File;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::sync_channel;
use std::sync::{Arc, Mutex};
use std::thread;

use half::f16;
use polars::io::parquet::metadata::FileMetadataRef;
use polars::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::converter::PARQUET_ROW_GROUP_ROWS;
use crate::{
    COMPACT_POLICY_SIZE, PACKED_PLANE_BYTES, PackedBatchData, PrefetchMessage,
    PrefetchedPackedBatchIterator,
};

const VALUE_TYPE_COUNT: usize = 6;
const VALUE_FIELDS: usize = 3;
const VALUE_COUNT: usize = VALUE_TYPE_COUNT * VALUE_FIELDS;
const ROOT_VALUE_INDEX: usize = 4;

struct ParquetBatchIterator {
    path: PathBuf,
    batch_size: usize,
    rows: usize,
    row_group_offset: usize,
    frame: Option<DataFrame>,
    frame_offset: usize,
    metadata: FileMetadataRef,
    retention_numerator: usize,
    retention_denominator: usize,
    retention_seed: u64,
}

impl ParquetBatchIterator {
    fn open(
        path: PathBuf,
        batch_size: usize,
        retention_numerator: usize,
        retention_denominator: usize,
    ) -> PyResult<Self> {
        validate_retention(retention_numerator, retention_denominator)?;
        let mut reader = ParquetReader::new(File::open(&path).map_err(io_error)?);
        let rows = reader.num_rows().map_err(polars_error)?;
        let metadata = reader.get_metadata().map_err(polars_error)?.clone();
        let retention_seed = shard_seed(&path);
        Ok(Self {
            path,
            batch_size,
            rows,
            row_group_offset: 0,
            frame: None,
            frame_offset: 0,
            metadata,
            retention_numerator,
            retention_denominator,
            retention_seed,
        })
    }

    fn next_batch_data(&mut self) -> PyResult<Option<PackedBatchData>> {
        while self
            .frame
            .as_ref()
            .is_none_or(|frame| self.frame_offset + self.batch_size > frame.height())
        {
            let leftover = self.frame.as_ref().and_then(|frame| {
                (self.frame_offset < frame.height()).then(|| {
                    frame.slice(self.frame_offset as i64, frame.height() - self.frame_offset)
                })
            });
            if !self.load_row_group()? {
                return Ok(None);
            }
            if let Some(mut leftover) = leftover {
                leftover
                    .vstack_mut(self.frame.as_ref().expect("row group was loaded"))
                    .map_err(polars_error)?;
                self.frame = Some(leftover);
            }
        }
        let Some(frame) = &self.frame else {
            return Ok(None);
        };
        if self.frame_offset + self.batch_size > frame.height() {
            return Ok(None);
        }
        let batch = frame.slice(self.frame_offset as i64, self.batch_size);
        self.frame_offset += self.batch_size;
        build_batch_data(&batch).map(Some)
    }

    fn load_row_group(&mut self) -> PyResult<bool> {
        self.frame = None;
        self.frame_offset = 0;
        if self.row_group_offset >= self.rows {
            return Ok(false);
        }
        let row_count = PARQUET_ROW_GROUP_ROWS.min(self.rows - self.row_group_offset);
        let mut reader = ParquetReader::new(File::open(&self.path).map_err(io_error)?)
            .with_slice(Some((self.row_group_offset, row_count)));
        reader.set_metadata(self.metadata.clone());
        let frame = reader.finish().map_err(polars_error)?;
        self.frame = Some(if self.retention_numerator == self.retention_denominator {
            frame
        } else {
            let retained = (0..row_count)
                .map(|offset| {
                    retain_row(
                        self.row_group_offset + offset,
                        self.retention_seed,
                        self.retention_numerator,
                        self.retention_denominator,
                    )
                })
                .collect::<Vec<_>>();
            let mask = BooleanChunked::from_slice("retained".into(), &retained);
            frame.filter(&mask).map_err(polars_error)?
        });
        self.row_group_offset += row_count;
        Ok(true)
    }
}

pub(crate) fn iter_prefetched_parquet_batches(
    paths: Vec<PathBuf>,
    batch_size: usize,
    prefetch_per_thread: usize,
    threads: usize,
    retention_numerator: usize,
    retention_denominator: usize,
) -> PyResult<PrefetchedPackedBatchIterator> {
    if prefetch_per_thread == 0 {
        return Err(PyValueError::new_err(
            "prefetch_per_thread must be positive",
        ));
    }
    if threads == 0 {
        return Err(PyValueError::new_err("threads must be positive"));
    }
    validate_retention(retention_numerator, retention_denominator)?;

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
                let Some(path) = worker_paths
                    .lock()
                    .expect("paths mutex poisoned")
                    .pop_front()
                else {
                    let _ = sender.send(PrefetchMessage::End);
                    break;
                };
                let mut iterator = match ParquetBatchIterator::open(
                    path,
                    batch_size,
                    retention_numerator,
                    retention_denominator,
                ) {
                    Ok(iterator) => iterator,
                    Err(error) => {
                        let _ = sender.send(PrefetchMessage::Error(error.to_string()));
                        return;
                    }
                };
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

pub(crate) fn parquet_retention_counts(
    path: PathBuf,
    batch_size: usize,
) -> PyResult<(usize, usize, usize, usize, usize, usize, usize)> {
    let mut reader = ParquetReader::new(File::open(&path).map_err(io_error)?);
    let rows = reader.num_rows().map_err(polars_error)?;
    let seed = shard_seed(&path);
    let quarter = retained_row_count(rows, seed, 1, 4);
    let half = retained_row_count(rows, seed, 2, 4);
    Ok((
        rows,
        rows,
        half,
        quarter,
        rows / batch_size * batch_size,
        half / batch_size * batch_size,
        quarter / batch_size * batch_size,
    ))
}

fn validate_retention(numerator: usize, denominator: usize) -> PyResult<()> {
    if denominator == 0 || numerator == 0 || numerator > denominator {
        return Err(PyValueError::new_err(
            "retention fraction must satisfy 0 < numerator <= denominator",
        ));
    }
    Ok(())
}

fn retain_row(index: usize, seed: u64, numerator: usize, denominator: usize) -> bool {
    let mut value = seed ^ (index as u64).wrapping_mul(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    ((value ^ (value >> 31)) % denominator as u64) < numerator as u64
}

fn retained_row_count(rows: usize, seed: u64, numerator: usize, denominator: usize) -> usize {
    (0..rows)
        .filter(|index| retain_row(*index, seed, numerator, denominator))
        .count()
}

fn shard_seed(path: &std::path::Path) -> u64 {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    let hash = name.bytes().fold(0xcbf29ce484222325_u64, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
    });
    hash
}

fn build_batch_data(frame: &DataFrame) -> PyResult<PackedBatchData> {
    let records = frame.height();
    let planes = frame
        .column("planes")
        .map_err(polars_error)?
        .binary()
        .map_err(polars_error)?;
    let castling = frame
        .column("castling")
        .map_err(polars_error)?
        .u8()
        .map_err(polars_error)?;
    let side_to_move = frame
        .column("side_to_move")
        .map_err(polars_error)?
        .bool()
        .map_err(polars_error)?;
    let rule50 = frame
        .column("rule50")
        .map_err(polars_error)?
        .u8()
        .map_err(polars_error)?;
    let policy_indices = frame
        .column("policy_indices")
        .map_err(polars_error)?
        .list()
        .map_err(polars_error)?;
    let policy_probs = frame
        .column("policy_probs_f16")
        .map_err(polars_error)?
        .list()
        .map_err(polars_error)?;
    let root_q = frame
        .column("root_q")
        .map_err(polars_error)?
        .f32()
        .map_err(polars_error)?;
    let root_d = frame
        .column("root_d")
        .map_err(polars_error)?
        .f32()
        .map_err(polars_error)?;
    let root_m = frame
        .column("root_m")
        .map_err(polars_error)?
        .f32()
        .map_err(polars_error)?;

    let mut packed_planes = vec![0_u8; records * PACKED_PLANE_BYTES];
    let mut scalars = vec![0.0_f32; records * 8];
    let mut output_indices = vec![-1_i16; records * COMPACT_POLICY_SIZE];
    let mut output_probs = vec![f16::ZERO; records * COMPACT_POLICY_SIZE];
    let mut value = vec![0.0_f32; records * VALUE_COUNT];

    let index_rows = policy_indices.amortized_iter();
    let probability_rows = policy_probs.amortized_iter();
    for (row, (indices, probabilities)) in index_rows.zip(probability_rows).enumerate() {
        let plane = planes
            .get(row)
            .ok_or_else(|| PyValueError::new_err("planes contains null values"))?;
        if plane.len() != PACKED_PLANE_BYTES {
            return Err(PyValueError::new_err(format!(
                "planes row has {} bytes, expected {PACKED_PLANE_BYTES}",
                plane.len()
            )));
        }
        let plane_start = row * PACKED_PLANE_BYTES;
        packed_planes[plane_start..plane_start + PACKED_PLANE_BYTES].copy_from_slice(plane);

        let flags = required(castling.get(row), "castling")?;
        let scalar_start = row * 8;
        scalars[scalar_start] = (flags & 1) as f32;
        scalars[scalar_start + 1] = ((flags >> 1) & 1) as f32;
        scalars[scalar_start + 2] = ((flags >> 2) & 1) as f32;
        scalars[scalar_start + 3] = ((flags >> 3) & 1) as f32;
        scalars[scalar_start + 4] = required(side_to_move.get(row), "side_to_move")? as u8 as f32;
        scalars[scalar_start + 5] = required(rule50.get(row), "rule50")? as f32;
        scalars[scalar_start + 7] = 1.0;

        let indices =
            indices.ok_or_else(|| PyValueError::new_err("policy_indices contains null"))?;
        let probabilities =
            probabilities.ok_or_else(|| PyValueError::new_err("policy_probs_f16 contains null"))?;
        let indices = indices.as_ref().u16().map_err(polars_error)?;
        let probabilities = probabilities.as_ref().u16().map_err(polars_error)?;
        if indices.len() != probabilities.len() || indices.len() > COMPACT_POLICY_SIZE {
            return Err(PyValueError::new_err("invalid compact policy row"));
        }
        let output_start = row * COMPACT_POLICY_SIZE;
        for (offset, (index, probability)) in indices
            .into_no_null_iter()
            .zip(probabilities.into_no_null_iter())
            .enumerate()
        {
            output_indices[output_start + offset] = index as i16;
            output_probs[output_start + offset] = f16::from_bits(probability);
        }

        let value_start = row * VALUE_COUNT + ROOT_VALUE_INDEX * VALUE_FIELDS;
        value[value_start] = required(root_q.get(row), "root_q")?;
        value[value_start + 1] = required(root_d.get(row), "root_d")?;
        value[value_start + 2] = required(root_m.get(row), "root_m")?;
        value[row * VALUE_COUNT + VALUE_COUNT - 1] = f32::NAN;
    }

    Ok(PackedBatchData {
        records,
        packed_planes,
        scalars,
        policy_indices: output_indices,
        policy_probs: output_probs,
        value,
    })
}

fn required<T>(value: Option<T>, name: &str) -> PyResult<T> {
    value.ok_or_else(|| PyValueError::new_err(format!("{name} contains null values")))
}

fn io_error(error: std::io::Error) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn polars_error(error: PolarsError) -> PyErr {
    PyValueError::new_err(error.to_string())
}
