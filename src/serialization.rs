//! Translated from velocyto/serialization.py
//! Uses hdf5-pure-rust instead of h5py, flate2 instead of zlib.

use flate2::read::ZlibDecoder;
use flate2::write::ZlibEncoder;
use flate2::Compression;
use hdf5_pure_rust::{File, WritableFile};
use ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::io::{Read, Write};

/// A typed variant of an HDF5 dataset value.
pub enum HdfValue {
    Array1F64(ndarray::Array1<f64>),
    Array2F64(ndarray::Array2<f64>),
    Array1U16(ndarray::Array1<u16>),
    Array2U16(ndarray::Array2<u16>),
    Bytes(Vec<u8>),
}

/// Compresses bytes with zlib. NOTE: Unlike Python's `_obj2uint`, this does NOT pickle first — round-trips with Python's `_uint2obj` require external pickle handling.
pub fn obj2uint(data: &[u8], compression: u32) -> Vec<u8> {
    let level = Compression::new(compression.min(9));
    let mut encoder = ZlibEncoder::new(Vec::new(), level);
    encoder.write_all(data).expect("zlib compress write failed");
    encoder.finish().expect("zlib compress finish failed")
}

/// Decompresses zlib-compressed bytes. NOTE: Unlike Python's `_uint2obj`, this does NOT unpickle — round-trips with Python's `_obj2uint` require external pickle handling.
pub fn uint2obj(data: &[u8]) -> anyhow::Result<Vec<u8>> {
    let mut decoder = ZlibDecoder::new(data);
    let mut out = Vec::new();
    decoder.read_to_end(&mut out)?;
    Ok(out)
}

/// Writes an HDF5 file with the given datasets and attributes.
///
/// - `HdfValue::Array*` → chunked, deflate-compressed numeric dataset (key as-is)
/// - `HdfValue::Bytes`  → zlib-compress then write as u8 dataset (key prefixed `"&"`)
pub fn dump_hdf5(
    filename: &str,
    attrs: &HashMap<String, HdfValue>,
    data_compression: u32,
    chunks: (usize, usize),
    noarray_compression: u32,
) -> anyhow::Result<()> {
    // Remove file if it exists (mirror Python behaviour)
    if std::path::Path::new(filename).exists() {
        std::fs::remove_file(filename)?;
    }

    let mut wf =
        WritableFile::create(filename).map_err(|e| anyhow::anyhow!("HDF5 create error: {e:?}"))?;

    for (key, value) in attrs {
        match value {
            HdfValue::Array1F64(arr) => {
                let n = arr.len();
                let chunk = (chunks.0.min(n)).max(1) as u64;
                wf.new_dataset_builder(key)
                    .shape(&[n as u64])
                    .chunk(&[chunk])
                    .deflate(data_compression)
                    .write::<f64>(arr.as_slice().unwrap())
                    .map_err(|e| anyhow::anyhow!("HDF5 write Array1F64 '{key}': {e:?}"))?;
            }
            HdfValue::Array2F64(arr) => {
                let (r, c) = arr.dim();
                let chunk0 = (chunks.0.min(r)).max(1) as u64;
                let chunk1 = (chunks.1.min(c)).max(1) as u64;
                wf.new_dataset_builder(key)
                    .shape(&[r as u64, c as u64])
                    .chunk(&[chunk0, chunk1])
                    .deflate(data_compression)
                    .write::<f64>(arr.as_slice().unwrap())
                    .map_err(|e| anyhow::anyhow!("HDF5 write Array2F64 '{key}': {e:?}"))?;
            }
            HdfValue::Array1U16(arr) => {
                let n = arr.len();
                let chunk = (chunks.0.min(n)).max(1) as u64;
                wf.new_dataset_builder(key)
                    .shape(&[n as u64])
                    .chunk(&[chunk])
                    .deflate(data_compression)
                    .write::<u16>(arr.as_slice().unwrap())
                    .map_err(|e| anyhow::anyhow!("HDF5 write Array1U16 '{key}': {e:?}"))?;
            }
            HdfValue::Array2U16(arr) => {
                let (r, c) = arr.dim();
                let chunk0 = (chunks.0.min(r)).max(1) as u64;
                let chunk1 = (chunks.1.min(c)).max(1) as u64;
                wf.new_dataset_builder(key)
                    .shape(&[r as u64, c as u64])
                    .chunk(&[chunk0, chunk1])
                    .deflate(data_compression)
                    .write::<u16>(arr.as_slice().unwrap())
                    .map_err(|e| anyhow::anyhow!("HDF5 write Array2U16 '{key}': {e:?}"))?;
            }
            HdfValue::Bytes(bytes) => {
                // Compress with zlib then store as u8 dataset under "&key"
                let compressed = obj2uint(bytes, noarray_compression);
                let n = compressed.len();
                let chunk = (1024usize.min(n)).max(1) as u64;
                let ds_key = format!("&{key}");
                wf.new_dataset_builder(&ds_key)
                    .shape(&[n as u64])
                    .chunk(&[chunk])
                    .deflate(data_compression)
                    .write::<u8>(&compressed)
                    .map_err(|e| anyhow::anyhow!("HDF5 write Bytes '{key}': {e:?}"))?;
            }
        }
    }

    wf.flush()
        .map_err(|e| anyhow::anyhow!("HDF5 flush error: {e:?}"))?;
    Ok(())
}

/// Reads an HDF5 file and returns its contents as a HashMap of string keys to HdfValue variants.
///
/// Datasets whose name starts with `"&"` are zlib-decompressed and returned
/// as `HdfValue::Bytes`. All others are returned as their native numeric type
/// (currently u16 2-D arrays for loom layer matrices, else raw bytes).
pub fn load_hdf5(filename: &str) -> anyhow::Result<HashMap<String, HdfValue>> {
    let file = File::open(filename).map_err(|e| anyhow::anyhow!("HDF5 open error: {e:?}"))?;

    let mut result = HashMap::new();

    // Collect all dataset names at root level
    let names = file
        .member_names()
        .map_err(|e| anyhow::anyhow!("HDF5 member_names error: {e:?}"))?;

    for name in names {
        let ds = file
            .dataset(&name)
            .map_err(|e| anyhow::anyhow!("HDF5 dataset '{name}' error: {e:?}"))?;

        if name.starts_with('&') {
            // Bytes dataset: read u8, then zlib-decompress
            let raw: Vec<u8> = ds
                .read::<u8>()
                .map_err(|e| anyhow::anyhow!("HDF5 read bytes '{name}': {e:?}"))?;
            let decompressed = uint2obj(&raw)?;
            let key = name[1..].to_string();
            result.insert(key, HdfValue::Bytes(decompressed));
        } else {
            // Numeric dataset — infer from shape whether 1D or 2D.
            let shape = ds
                .shape()
                .map_err(|e| anyhow::anyhow!("HDF5 shape '{name}': {e:?}"))?;
            match shape.len() {
                1 => {
                    // Try f64 first, fall back to u16
                    if let Ok(arr) = ds.read_1d::<f64>() {
                        result.insert(name, HdfValue::Array1F64(arr));
                    } else {
                        let arr = ds
                            .read_1d::<u16>()
                            .map_err(|e| anyhow::anyhow!("HDF5 read_1d u16 '{name}': {e:?}"))?;
                        result.insert(name, HdfValue::Array1U16(arr));
                    }
                }
                2 => {
                    if let Ok(arr) = ds.read_2d::<f64>() {
                        result.insert(name, HdfValue::Array2F64(arr));
                    } else {
                        let arr = ds
                            .read_2d::<u16>()
                            .map_err(|e| anyhow::anyhow!("HDF5 read_2d u16 '{name}': {e:?}"))?;
                        result.insert(name, HdfValue::Array2U16(arr));
                    }
                }
                _ => {
                    // Higher-rank or scalar: store raw bytes
                    let raw = ds
                        .read_raw()
                        .map_err(|e| anyhow::anyhow!("HDF5 read_raw '{name}': {e:?}"))?;
                    result.insert(name, HdfValue::Bytes(raw));
                }
            }
        }
    }

    Ok(result)
}
