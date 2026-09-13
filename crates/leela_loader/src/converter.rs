use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use half::f16;
use polars::io::parquet::write::BatchedWriter;
use polars::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::{
    BEST_D_OFFSET, BEST_M_OFFSET, BEST_Q_OFFSET, CASTLING_THEM_OO_OFFSET, CASTLING_THEM_OOO_OFFSET,
    CASTLING_US_OO_OFFSET, CASTLING_US_OOO_OFFSET, PACKED_PLANE_BYTES, PLANES_OFFSET,
    PLIES_LEFT_OFFSET, POLICY_OFFSET, POLICY_SIZE, RECORD_SIZE, RESULT_D_OFFSET, RESULT_Q_OFFSET,
    ROOT_D_OFFSET, ROOT_M_OFFSET, ROOT_Q_OFFSET, RULE50_OFFSET, SIDE_TO_MOVE_OFFSET,
    SimpleTarReader, read_f32, read_u32,
};

pub(crate) const PARQUET_ROW_GROUP_ROWS: usize = 65_536;
const ICEBERG_BUFFER_ROWS: usize = 32_768;
const CURRENT_POSITION_PIECE_PLANES: usize = 12;

#[derive(Default)]
struct Rows {
    planes: Vec<Vec<u8>>,
    castling: Vec<u8>,
    side_to_move: Vec<bool>,
    rule50: Vec<u8>,
    policy_indices: Vec<Vec<u16>>,
    policy_probs_f16: Vec<Vec<u16>>,
    root_q: Vec<f32>,
    root_d: Vec<f32>,
    root_m: Vec<f32>,
}

struct IcebergRows {
    source_archive: String,
    planes: Vec<Vec<u8>>,
    castling: Vec<i32>,
    side_to_move: Vec<bool>,
    rule50: Vec<i32>,
    policy_indices: Vec<Vec<u8>>,
    policy_probs_f16: Vec<Vec<u8>>,
    root_q: Vec<f32>,
    root_d: Vec<f32>,
    root_m: Vec<f32>,
    best_q: Vec<f32>,
    best_d: Vec<f32>,
    best_m: Vec<f32>,
    result_q: Vec<f32>,
    result_d: Vec<f32>,
    plies_left: Vec<f32>,
    num_pieces: Vec<i32>,
}

impl IcebergRows {
    fn with_capacity(source_archive: &str, capacity: usize) -> Self {
        Self {
            source_archive: source_archive.to_owned(),
            planes: Vec::with_capacity(capacity),
            castling: Vec::with_capacity(capacity),
            side_to_move: Vec::with_capacity(capacity),
            rule50: Vec::with_capacity(capacity),
            policy_indices: Vec::with_capacity(capacity),
            policy_probs_f16: Vec::with_capacity(capacity),
            root_q: Vec::with_capacity(capacity),
            root_d: Vec::with_capacity(capacity),
            root_m: Vec::with_capacity(capacity),
            best_q: Vec::with_capacity(capacity),
            best_d: Vec::with_capacity(capacity),
            best_m: Vec::with_capacity(capacity),
            result_q: Vec::with_capacity(capacity),
            result_d: Vec::with_capacity(capacity),
            plies_left: Vec::with_capacity(capacity),
            num_pieces: Vec::with_capacity(capacity),
        }
    }

    fn len(&self) -> usize {
        self.planes.len()
    }

    fn push(&mut self, record: &[u8]) -> PyResult<()> {
        let version = read_u32(record, 0);
        if version != 6 {
            return Err(PyValueError::new_err(format!(
                "unsupported LCZero record version {version}"
            )));
        }

        let plane_bytes = &record[PLANES_OFFSET..PLANES_OFFSET + PACKED_PLANE_BYTES];
        self.planes.push(plane_bytes.to_vec());
        self.castling.push(i32::from(
            (record[CASTLING_US_OOO_OFFSET] & 1)
                | ((record[CASTLING_US_OO_OFFSET] & 1) << 1)
                | ((record[CASTLING_THEM_OOO_OFFSET] & 1) << 2)
                | ((record[CASTLING_THEM_OO_OFFSET] & 1) << 3),
        ));
        self.side_to_move.push(record[SIDE_TO_MOVE_OFFSET] != 0);
        self.rule50.push(i32::from(record[RULE50_OFFSET]));

        let mut indices = Vec::with_capacity(128);
        let mut probabilities = Vec::with_capacity(128);
        for policy_index in 0..POLICY_SIZE {
            let probability = read_f32(record, POLICY_OFFSET + policy_index * 4);
            if probability >= 0.0 {
                indices.extend_from_slice(&(policy_index as u16).to_le_bytes());
                probabilities
                    .extend_from_slice(&f16::from_f32(probability).to_bits().to_le_bytes());
            }
        }
        self.policy_indices.push(indices);
        self.policy_probs_f16.push(probabilities);
        self.root_q.push(read_f32(record, ROOT_Q_OFFSET));
        self.root_d.push(read_f32(record, ROOT_D_OFFSET));
        self.root_m.push(read_f32(record, ROOT_M_OFFSET));
        self.best_q.push(read_f32(record, BEST_Q_OFFSET));
        self.best_d.push(read_f32(record, BEST_D_OFFSET));
        self.best_m.push(read_f32(record, BEST_M_OFFSET));
        self.result_q.push(read_f32(record, RESULT_Q_OFFSET));
        self.result_d.push(read_f32(record, RESULT_D_OFFSET));
        self.plies_left.push(read_f32(record, PLIES_LEFT_OFFSET));
        self.num_pieces.push(
            plane_bytes[..CURRENT_POSITION_PIECE_PLANES * 8]
                .iter()
                .map(|byte| byte.count_ones() as i32)
                .sum(),
        );
        Ok(())
    }

    fn into_frame(self) -> PyResult<DataFrame> {
        let height = self.len();
        DataFrame::new(
            height,
            vec![
                BinaryChunked::from_iter_values(
                    "planes".into(),
                    self.planes.iter().map(Vec::as_slice),
                )
                .into_column(),
                Int32Chunked::from_vec("castling".into(), self.castling).into_column(),
                BooleanChunked::from_slice("side_to_move".into(), &self.side_to_move).into_column(),
                Int32Chunked::from_vec("rule50".into(), self.rule50).into_column(),
                BinaryChunked::from_iter_values(
                    "policy_indices".into(),
                    self.policy_indices.iter().map(Vec::as_slice),
                )
                .into_column(),
                BinaryChunked::from_iter_values(
                    "policy_probs_f16".into(),
                    self.policy_probs_f16.iter().map(Vec::as_slice),
                )
                .into_column(),
                Float32Chunked::from_vec("root_q".into(), self.root_q).into_column(),
                Float32Chunked::from_vec("root_d".into(), self.root_d).into_column(),
                Float32Chunked::from_vec("root_m".into(), self.root_m).into_column(),
                Float32Chunked::from_vec("best_q".into(), self.best_q).into_column(),
                Float32Chunked::from_vec("best_d".into(), self.best_d).into_column(),
                Float32Chunked::from_vec("best_m".into(), self.best_m).into_column(),
                Float32Chunked::from_vec("result_q".into(), self.result_q).into_column(),
                Float32Chunked::from_vec("result_d".into(), self.result_d).into_column(),
                Float32Chunked::from_vec("plies_left".into(), self.plies_left).into_column(),
                Int32Chunked::from_vec("num_pieces".into(), self.num_pieces).into_column(),
                StringChunked::from_iter_values(
                    "source_archive".into(),
                    std::iter::repeat_n(self.source_archive.as_str(), height),
                )
                .into_column(),
            ],
        )
        .map_err(polars_error)
    }
}

struct PartitionWriter {
    rows: IcebergRows,
    partial_path: PathBuf,
    final_path: PathBuf,
    writer: Option<BatchedWriter<File>>,
    records: usize,
}

impl PartitionWriter {
    fn new(source_archive: &str, output_dir: &Path, piece_partition: i32) -> PyResult<Self> {
        let partition_dir = output_dir.join(format!("num_pieces_trunc={piece_partition}"));
        std::fs::create_dir_all(&partition_dir).map_err(io_error)?;
        let stem = Path::new(source_archive)
            .file_stem()
            .and_then(|value| value.to_str())
            .ok_or_else(|| PyValueError::new_err("source archive has no UTF-8 file stem"))?;
        let final_path = partition_dir.join(format!("{stem}.parquet"));
        let partial_path = partition_dir.join(format!("{stem}.parquet.partial"));
        if final_path.exists() {
            return Err(PyValueError::new_err(format!(
                "output already exists: {}",
                final_path.display()
            )));
        }
        if partial_path.exists() {
            std::fs::remove_file(&partial_path).map_err(io_error)?;
        }
        Ok(Self {
            rows: IcebergRows::with_capacity(source_archive, ICEBERG_BUFFER_ROWS),
            partial_path,
            final_path,
            writer: None,
            records: 0,
        })
    }

    fn push(&mut self, record: &[u8]) -> PyResult<()> {
        self.rows.push(record)?;
        if self.rows.len() == ICEBERG_BUFFER_ROWS {
            self.flush()?;
        }
        Ok(())
    }

    fn flush(&mut self) -> PyResult<()> {
        if self.rows.len() == 0 {
            return Ok(());
        }
        let source_archive = self.rows.source_archive.clone();
        let rows = std::mem::replace(
            &mut self.rows,
            IcebergRows::with_capacity(&source_archive, ICEBERG_BUFFER_ROWS),
        );
        let records = rows.len();
        let frame = rows.into_frame()?;
        if self.writer.is_none() {
            let file = File::create(&self.partial_path).map_err(io_error)?;
            self.writer = Some(
                ParquetWriter::new(file)
                    .batched(frame.schema())
                    .map_err(polars_error)?,
            );
        }
        self.writer
            .as_mut()
            .expect("writer was initialized")
            .write_batch(&frame)
            .map_err(polars_error)?;
        self.records += records;
        Ok(())
    }

    fn finish(mut self) -> PyResult<(String, usize, u64)> {
        self.flush()?;
        self.writer
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("partition had no training records"))?
            .finish()
            .map_err(polars_error)?;
        std::fs::rename(&self.partial_path, &self.final_path).map_err(io_error)?;
        let bytes = self.final_path.metadata().map_err(io_error)?.len();
        Ok((
            self.final_path.to_string_lossy().into_owned(),
            self.records,
            bytes,
        ))
    }
}

impl Rows {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            planes: Vec::with_capacity(capacity),
            castling: Vec::with_capacity(capacity),
            side_to_move: Vec::with_capacity(capacity),
            rule50: Vec::with_capacity(capacity),
            policy_indices: Vec::with_capacity(capacity),
            policy_probs_f16: Vec::with_capacity(capacity),
            root_q: Vec::with_capacity(capacity),
            root_d: Vec::with_capacity(capacity),
            root_m: Vec::with_capacity(capacity),
        }
    }

    fn len(&self) -> usize {
        self.planes.len()
    }

    fn push(&mut self, record: &[u8]) -> PyResult<()> {
        let version = read_u32(record, 0);
        if version != 6 {
            return Err(PyValueError::new_err(format!(
                "unsupported LCZero record version {version}"
            )));
        }
        self.planes
            .push(record[PLANES_OFFSET..PLANES_OFFSET + PACKED_PLANE_BYTES].to_vec());
        self.castling.push(
            (record[CASTLING_US_OOO_OFFSET] & 1)
                | ((record[CASTLING_US_OO_OFFSET] & 1) << 1)
                | ((record[CASTLING_THEM_OOO_OFFSET] & 1) << 2)
                | ((record[CASTLING_THEM_OO_OFFSET] & 1) << 3),
        );
        self.side_to_move.push(record[SIDE_TO_MOVE_OFFSET] != 0);
        self.rule50.push(record[RULE50_OFFSET]);

        let mut indices = Vec::with_capacity(64);
        let mut probabilities = Vec::with_capacity(64);
        for policy_index in 0..POLICY_SIZE {
            let probability = read_f32(record, POLICY_OFFSET + policy_index * 4);
            if probability >= 0.0 {
                indices.push(policy_index as u16);
                probabilities.push(f16::from_f32(probability).to_bits());
            }
        }
        self.policy_indices.push(indices);
        self.policy_probs_f16.push(probabilities);
        self.root_q.push(read_f32(record, ROOT_Q_OFFSET));
        self.root_d.push(read_f32(record, ROOT_D_OFFSET));
        self.root_m.push(read_f32(record, ROOT_M_OFFSET));
        Ok(())
    }

    fn into_frame(self) -> PyResult<DataFrame> {
        let height = self.len();
        let mut indices = ListPrimitiveChunkedBuilder::<UInt16Type>::new(
            "policy_indices".into(),
            height,
            self.policy_indices.iter().map(Vec::len).sum(),
            DataType::UInt16,
        );
        let mut probabilities = ListPrimitiveChunkedBuilder::<UInt16Type>::new(
            "policy_probs_f16".into(),
            height,
            self.policy_probs_f16.iter().map(Vec::len).sum(),
            DataType::UInt16,
        );
        for values in &self.policy_indices {
            indices.append_slice(values);
        }
        for values in &self.policy_probs_f16 {
            probabilities.append_slice(values);
        }

        DataFrame::new(
            height,
            vec![
                BinaryChunked::from_iter_values(
                    "planes".into(),
                    self.planes.iter().map(Vec::as_slice),
                )
                .into_column(),
                UInt8Chunked::from_vec("castling".into(), self.castling).into_column(),
                BooleanChunked::from_slice("side_to_move".into(), &self.side_to_move).into_column(),
                UInt8Chunked::from_vec("rule50".into(), self.rule50).into_column(),
                indices.finish().into_column(),
                probabilities.finish().into_column(),
                Float32Chunked::from_vec("root_q".into(), self.root_q).into_column(),
                Float32Chunked::from_vec("root_d".into(), self.root_d).into_column(),
                Float32Chunked::from_vec("root_m".into(), self.root_m).into_column(),
            ],
        )
        .map_err(polars_error)
    }
}

pub(crate) fn convert_lc0_tar_to_parquet(
    input: PathBuf,
    output: PathBuf,
) -> PyResult<(usize, u64, u64)> {
    let input_bytes = input.metadata().map_err(io_error)?.len();
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).map_err(io_error)?;
    }

    let mut reader = SimpleTarReader::open(input)?;
    let mut rows = Rows::with_capacity(PARQUET_ROW_GROUP_ROWS);
    let mut writer: Option<BatchedWriter<File>> = None;
    let mut records = 0;
    while let Some(payload) = reader.next_regular_payload()? {
        if payload.len() % RECORD_SIZE != 0 {
            return Err(PyValueError::new_err(format!(
                "LCZero chunk has {} bytes, not a multiple of {RECORD_SIZE}",
                payload.len()
            )));
        }
        for record in payload.chunks_exact(RECORD_SIZE) {
            rows.push(record)?;
            if rows.len() == PARQUET_ROW_GROUP_ROWS {
                records += write_rows(
                    std::mem::replace(&mut rows, Rows::with_capacity(PARQUET_ROW_GROUP_ROWS)),
                    &output,
                    &mut writer,
                )?;
            }
        }
    }
    if rows.len() > 0 {
        records += write_rows(rows, &output, &mut writer)?;
    }
    let writer = writer.ok_or_else(|| PyValueError::new_err("input had no training records"))?;
    writer.finish().map_err(polars_error)?;
    let output_bytes = output.metadata().map_err(io_error)?.len();
    Ok((records, input_bytes, output_bytes))
}

pub(crate) fn convert_lc0_tar_to_iceberg_parquet(
    input: PathBuf,
    output_dir: PathBuf,
) -> PyResult<(usize, u64, u64, Vec<(String, usize, u64)>)> {
    let input_bytes = input.metadata().map_err(io_error)?.len();
    let source_archive = input
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| PyValueError::new_err("source archive has no UTF-8 file name"))?
        .to_owned();
    let mut reader = SimpleTarReader::open(input)?;
    let mut partitions: HashMap<i32, PartitionWriter> = HashMap::new();
    let mut records = 0;

    while let Some(payload) = reader.next_regular_payload()? {
        if payload.len() % RECORD_SIZE != 0 {
            return Err(PyValueError::new_err(format!(
                "LCZero chunk has {} bytes, not a multiple of {RECORD_SIZE}",
                payload.len()
            )));
        }
        for record in payload.chunks_exact(RECORD_SIZE) {
            let plane_bytes = &record[PLANES_OFFSET..PLANES_OFFSET + PACKED_PLANE_BYTES];
            let num_pieces: i32 = plane_bytes[..CURRENT_POSITION_PIECE_PLANES * 8]
                .iter()
                .map(|byte| byte.count_ones() as i32)
                .sum();
            let piece_partition = num_pieces / 4 * 4;
            if !partitions.contains_key(&piece_partition) {
                partitions.insert(
                    piece_partition,
                    PartitionWriter::new(&source_archive, &output_dir, piece_partition)?,
                );
            }
            partitions
                .get_mut(&piece_partition)
                .expect("partition writer was initialized")
                .push(record)?;
            records += 1;
        }
    }
    if records == 0 {
        return Err(PyValueError::new_err("input had no training records"));
    }

    let mut files = partitions
        .into_values()
        .map(PartitionWriter::finish)
        .collect::<PyResult<Vec<_>>>()?;
    files.sort_by(|left, right| left.0.cmp(&right.0));
    let output_bytes = files.iter().map(|(_, _, bytes)| bytes).sum();
    Ok((records, input_bytes, output_bytes, files))
}

fn write_rows(
    rows: Rows,
    output: &Path,
    writer: &mut Option<BatchedWriter<File>>,
) -> PyResult<usize> {
    let records = rows.len();
    let frame = rows.into_frame()?;
    if writer.is_none() {
        let file = File::create(output).map_err(io_error)?;
        *writer = Some(
            ParquetWriter::new(file)
                .batched(frame.schema())
                .map_err(polars_error)?,
        );
    }
    writer
        .as_mut()
        .expect("writer was initialized")
        .write_batch(&frame)
        .map_err(polars_error)?;
    Ok(records)
}

fn io_error(error: std::io::Error) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn polars_error(error: PolarsError) -> PyErr {
    PyValueError::new_err(error.to_string())
}
