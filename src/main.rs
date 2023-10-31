use whisper_rs::{WhisperContext, FullParams, SamplingStrategy};
use std::{path::Path, cmp};
use std::fs::File;
use std::io;
use std::io::Write;
use std::io::Read;
use hound::{SampleFormat, WavReader};
use atty::Stream;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    model: String,

    #[arg(short, long)]
    input: Option<String>,

    #[arg(short, long)]
    output: String,
}

fn write_to_file(path: &Path, lines: Vec<String>) {
    let mut file = File::create(path).expect("Could not create file");
    for line in lines {
        file.write_all(line.as_bytes()).expect("Could not write to file");
    }
}

fn construct_wavreader(audio_file_path_raw: &Option<String>) -> WavReader<Box<dyn io::BufRead>> {
    if atty::is(Stream::Stdin) {
        let audio_file_path_resolved = audio_file_path_raw.as_ref().expect("audio file not provided");
        let audio_file_path = Path::new(audio_file_path_resolved);
        if !audio_file_path.exists() {
            panic!("audio file doesn't exist");
        }
        let file = File::open(audio_file_path).expect("Failed to open audio file");
        let buf_reader = io::BufReader::new(file);
        return WavReader::new(Box::new(buf_reader) as Box<dyn io::BufRead>).expect("failed to read file");
    } else {
        let stdin = io::stdin();
        let patched_reader = WavLengthPatcher::new(stdin);
        let buf_reader = io::BufReader::new(patched_reader);
        return WavReader::new(Box::new(buf_reader) as Box<dyn io::BufRead>).expect("Input is not valid wav");
    };
}

fn parse_wav(reader: WavReader<Box<dyn io::BufRead>>) -> Vec<i16> {
    if reader.spec().channels != 1 {
        panic!("expected mono audio file");
    }
    if reader.spec().sample_format != SampleFormat::Int {
        panic!("expected integer sample format");
    }
    if reader.spec().sample_rate != 16000 {
        panic!("expected 16KHz sample rate");
    }
    if reader.spec().bits_per_sample != 16 {
        panic!("expected 16 bits per sample");
    }

    reader
        .into_samples::<i16>()
        .map_while(|sample| match sample {
            Ok(sample) => {
                Some(sample)
            },
            Err(hound::Error::IoError(e)) => {
                if let Some(ee) = e.get_ref() {
                    if "Failed to read enough bytes." == &format!("{}", ee) {
                        return None
                    }
                }
                panic!("Error reading audio data: {:#?}", e);
            }
            Err(e) => panic!("Error reading audio data: {:#?}", e)
        })
        .collect::<Vec<_>>()
}

fn segment_time_to_srt_time_string(time: i64) -> String {
    let positive_time = cmp::max(0, time) * 10;
    let ms = positive_time % 1000;
    let seconds = (positive_time / 1000) % 60;
    let minutes = (positive_time / 1000 / 60) % 60;
    let hours = positive_time / 1000 / 60 / 60;
    format!("{:02}:{:02}:{:02},{:03}", hours, minutes, seconds, ms)
}

fn main() {
    let args = Args::parse();
    
    let audio_file_path_raw = &args.input;
    let model_path_raw = &args.model;
    let output_path_raw = &args.output;

    let model_path = Path::new(model_path_raw);
    if !model_path.exists() {
        panic!("model does not exist");
    }

    let wavreader = construct_wavreader(audio_file_path_raw);
    let audio_data = parse_wav(wavreader);
    let ingested_wav = whisper_rs::convert_integer_to_float_audio(&audio_data);

    let ctx = WhisperContext::new(&model_path.to_string_lossy()).expect("Failed to load model");

    let params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });

    let mut state = ctx.create_state().expect("failed to create state");
    state.full(params, &ingested_wav).expect("failed to run model");

    // fetch the results
    let num_segments = state.full_n_segments().expect("failed to get number of segments");

    println!("{}", num_segments);
    
    let mut srt_sequences: Vec<String> = Vec::new();
    let mut timestamped_lines: Vec<String> = Vec::new();

    for i in 0..num_segments {
        let segment = state.full_get_segment_text(i).expect("failed to get segment");
        let start_timestamp = state.full_get_segment_t0(i).expect("failed to get segment start timestamp");
        let end_timestamp = state.full_get_segment_t1(i).expect("failed to get segment end timestamp");

        let srt_start_timestamp = segment_time_to_srt_time_string(start_timestamp);
        let srt_end_timestamp = segment_time_to_srt_time_string(end_timestamp);
        let srt_formatted: String = format!("{}\n{srt_start_timestamp} --> {srt_end_timestamp}\n{segment}\n\n", i+1);
        srt_sequences.push(srt_formatted);

        let timestamped: String = format!("[{} - {}]: {}", start_timestamp, end_timestamp, segment);
        println!("{}", timestamped);
        timestamped_lines.push(format!("{}\n", timestamped));
    }

    write_to_file(Path::new(&format!("{}.txt", output_path_raw)), timestamped_lines);
    write_to_file(Path::new(&format!("{}.srt", output_path_raw)), srt_sequences);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn _segment_time_to_srt_time_string() {
        assert_eq!(segment_time_to_srt_time_string(1999), "00:00:19,990");
        assert_eq!(segment_time_to_srt_time_string(838850), "02:19:48,500");
        assert_eq!(segment_time_to_srt_time_string(5602555), "15:33:45,550");
        assert_eq!(segment_time_to_srt_time_string(-4550), "00:00:00,000");
    }
}


















struct WavLengthPatcher<R: Read> {
    stream: R,
    pos: u8,
}

impl <R: Read> WavLengthPatcher<R> {
    fn new(stream: R) -> Self {
        Self {
            stream,
            pos: 0,
        }
    }
}

impl <R: Read> Read for WavLengthPatcher<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let l = self.stream.read(buf)?;
        if self.pos < 0x4e {
            // fake bytes @ 0x04 and 0x4A from 0xFF -> 0xF0 (zero lowest 4 bits of file & data chunk length)
            let capped_l = l.min(0x4e) as u8;
            for p in 0 .. capped_l {
                match p + self.pos {
                    0x04 | 0x4A => {
                        assert_eq!(buf[p as usize], 0xFF);
                        buf[p as usize] = 0xF0;
                    }
                    0x05 | 0x06 | 0x07 | 0x4b | 0x4c | 0x4d => {
                        assert_eq!(buf[p as usize], 0xFF);
                    }
                    _ => ()
                }
            }
            self.pos += capped_l; // cannot overflow
        }
        Ok(l)
    }
}
