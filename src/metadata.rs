//! Translated from velocyto/metadata.py

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

/// A single row of typed metadata from a CSV file. Fields are stored as strings in a HashMap.
pub struct Metadata {
    pub types: HashMap<String, String>,
    pub dict: HashMap<String, String>,
}

impl Metadata {
    /// Creates a Metadata row from a map of field names to string values.
    pub fn new(keys: Vec<String>, values: Vec<String>, types: Vec<String>) -> Self {
        let type_map: HashMap<String, String> =
            keys.iter().cloned().zip(types.into_iter()).collect();
        let dict: HashMap<String, String> = keys.into_iter().zip(values.into_iter()).collect();
        Metadata {
            types: type_map,
            dict,
        }
    }

    /// Returns the value of a metadata field by name.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.dict.get(key).map(|s| s.as_str())
    }
}

/// A collection of typed metadata rows loaded from a CSV file.
pub struct MetadataCollection {
    pub items: Vec<Metadata>,
}

impl MetadataCollection {
    /// Creates a new empty MetadataCollection.
    pub fn new(filename: &str) -> anyhow::Result<Self> {
        let mut mc = MetadataCollection { items: Vec::new() };
        mc.load(filename)?;
        Ok(mc)
    }

    /// Loads a CSV file as a MetadataCollection. Auto-detects delimiter (comma or tab). First row is treated as a header.
    pub fn load(&mut self, filename: &str) -> anyhow::Result<()> {
        let file = File::open(filename)?;
        let reader = BufReader::new(file);

        // Sniff delimiter by reading first non-empty line
        let mut lines: Vec<String> = reader
            .lines()
            .filter_map(|l| l.ok())
            .filter(|l| !l.trim().is_empty())
            .collect();

        if lines.is_empty() {
            return Ok(());
        }

        // Detect delimiter from header line
        let delimiter = detect_delimiter(&lines[0]);

        let mut keys: Option<Vec<String>> = None;
        let mut types: Option<Vec<String>> = None;

        for line in &lines {
            if line.trim().is_empty() {
                continue;
            }
            let row: Vec<String> = split_line(line, delimiter);
            if row.is_empty() {
                continue;
            }

            if keys.is_none() {
                // Check if first column has "key:type" format
                if row[0].contains(':') && row[0].split(':').count() == 2 {
                    let ks: Vec<String> = row
                        .iter()
                        .map(|r| r.splitn(2, ':').next().unwrap_or("").to_string())
                        .collect();
                    let ts: Vec<String> = row
                        .iter()
                        .map(|r| r.splitn(2, ':').nth(1).unwrap_or("None").to_string())
                        .collect();
                    keys = Some(ks);
                    types = Some(ts);
                } else {
                    let ts = vec!["None".to_string(); row.len()];
                    keys = Some(row);
                    types = Some(ts);
                }
            } else {
                let ks = keys.as_ref().unwrap().clone();
                let ts = types.as_ref().unwrap().clone();
                self.items.push(Metadata::new(ks, row, ts));
            }
        }

        Ok(())
    }

    /// Filters rows where the given field equals the given value. Corresponds to Python's `where()` method.
    pub fn where_eq(&self, key: &str, value: &str) -> Vec<&Metadata> {
        self.items
            .iter()
            .filter(|item| item.get(key) == Some(value))
            .collect()
    }
}

fn detect_delimiter(line: &str) -> char {
    // Try common delimiters: comma, tab, semicolon
    let candidates = [',', '\t', ';'];
    let mut best = ',';
    let mut best_count = 0usize;
    for &c in &candidates {
        let count = line.matches(c).count();
        if count > best_count {
            best_count = count;
            best = c;
        }
    }
    best
}

fn split_line(line: &str, delimiter: char) -> Vec<String> {
    line.split(delimiter)
        .map(|s| s.trim().to_string())
        .collect()
}
