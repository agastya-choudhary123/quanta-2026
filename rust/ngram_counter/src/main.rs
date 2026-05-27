use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;

use rayon::prelude::*;
use serde::Deserialize;

const MIN_N: usize = 3;
const MAX_N: usize = 7;
const MIN_TOP_COUNT: u32 = 2;

#[derive(Deserialize)]
struct Question {
    #[serde(default)]
    page: Option<String>,
    #[serde(default)]
    answer: Option<String>,
    #[serde(default)]
    full_question: Option<String>,
    #[serde(default)]
    text: Option<String>,
}

fn tokenize(s: &str) -> Vec<&str> {
    let mut words = Vec::new();
    let mut start: Option<usize> = None;
    for (i, c) in s.char_indices() {
        let is_alpha = c.is_ascii_alphabetic() || c == '\'';
        match (is_alpha, start) {
            (true, None) => start = Some(i),
            (false, Some(s_idx)) => {
                words.push(&s[s_idx..i]);
                start = None;
            }
            _ => {}
        }
    }
    if let Some(s_idx) = start {
        words.push(&s[s_idx..]);
    }
    words
}

fn extract_question(q: &Question) -> HashMap<String, HashMap<String, u32>> {
    let answer = q.page.as_deref().filter(|s| !s.is_empty())
        .or_else(|| q.answer.as_deref().filter(|s| !s.is_empty()));
    let answer = match answer {
        Some(a) => a,
        None => return HashMap::new(),
    };
    let text = q.full_question.as_deref().filter(|s| !s.is_empty())
        .or_else(|| q.text.as_deref().filter(|s| !s.is_empty()));
    let text = match text {
        Some(t) => t,
        None => return HashMap::new(),
    };

    let lower = text.to_lowercase();
    let words = tokenize(&lower);

    let mut map: HashMap<String, HashMap<String, u32>> = HashMap::new();
    for n in MIN_N..=MAX_N {
        if words.len() < n {
            break;
        }
        for i in 0..=(words.len() - n) {
            let phrase = words[i..i + n].join(" ");
            *map.entry(phrase).or_default().entry(answer.to_string()).or_insert(0) += 1;
        }
    }
    map
}

fn merge_into(mut base: HashMap<String, HashMap<String, u32>>, other: HashMap<String, HashMap<String, u32>>) -> HashMap<String, HashMap<String, u32>> {
    for (phrase, ans_counts) in other {
        let entry = base.entry(phrase).or_default();
        for (ans, cnt) in ans_counts {
            *entry.entry(ans).or_insert(0) += cnt;
        }
    }
    base
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: ngram_counter <input.jsonl> <output.json>");
        std::process::exit(1);
    }
    let input_path = PathBuf::from(&args[1]);
    let output_path = PathBuf::from(&args[2]);

    eprintln!("Reading {}...", input_path.display());
    let file = File::open(&input_path).expect("Cannot open input");
    let reader = BufReader::new(file);
    let lines: Vec<String> = reader.lines().filter_map(|l| l.ok()).collect();
    eprintln!("Loaded {} lines", lines.len());

    eprintln!("Extracting n-grams in parallel...");
    let global: HashMap<String, HashMap<String, u32>> = lines
        .par_iter()
        .filter_map(|line| serde_json::from_str::<Question>(line).ok())
        .map(|q| extract_question(&q))
        .reduce(HashMap::new, |a, b| merge_into(a, b));

    eprintln!("{} unique phrases before filter", global.len());

    let filtered: HashMap<&str, &HashMap<String, u32>> = global
        .iter()
        .filter(|(_, counts)| counts.values().copied().max().unwrap_or(0) >= MIN_TOP_COUNT)
        .map(|(k, v)| (k.as_str(), v))
        .collect();

    eprintln!("{} phrases after filtering (top count >= {})", filtered.len(), MIN_TOP_COUNT);

    eprintln!("Writing {}...", output_path.display());
    let out_file = File::create(&output_path).expect("Cannot create output");
    let mut writer = BufWriter::new(out_file);
    serde_json::to_writer(&mut writer, &filtered).expect("JSON write failed");
    writer.flush().unwrap();

    let size_mb = output_path.metadata().unwrap().len() as f64 / 1e6;
    eprintln!("Done. Output: {:.1} MB", size_mb);
}
