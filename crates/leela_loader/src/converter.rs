use std::fs::File;
use std::path::{Path, PathBuf};

use half::f16;
use polars::io::parquet::write::BatchedWriter;
use polars::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::{
    CASTLING_THEM_OO_OFFSET, CASTLING_THEM_OOO_OFFSET, CASTLING_US_OO_OFFSET,
    CASTLING_US_OOO_OFFSET, PACKED_PLANE_BYTES, PLANES_OFFSET, POLICY_OFFSET, POLICY_SIZE,
    RECORD_SIZE, ROOT_D_OFFSET, ROOT_M_OFFSET, ROOT_Q_OFFSET, RULE50_OFFSET, SIDE_TO_MOVE_OFFSET,
    SimpleTarReader, read_f32, read_u32,
};

pub(crate) const PARQUET_ROW_GROUP_ROWS: usize = 65_536;

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
