use whisper_rs::{WhisperContext, FullParams, SamplingStrategy};
use std::path::Path;
use hound::{SampleFormat, WavReader};

fn parse_wav_file(path: &Path) -> Vec<i16> {
    let reader = WavReader::open(path).expect("failed to read file");

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
        .map(|x| x.expect("sample"))
        .collect::<Vec<_>>()
}

fn main() {
    let audio_file_path = Path::new("./samples/test.wav");
    if !audio_file_path.exists() {
        panic!("audio file doesn't exist");
    }
    let audio_data = parse_wav_file(audio_file_path);
    let ingested_wav = whisper_rs::convert_integer_to_float_audio(&audio_data);

    let model_path = Path::new("./models/tiny.en.bin");
    if !model_path.exists() {
        panic!("model does not exist");
    }
    println!("{}", &model_path.to_string_lossy());
    let ctx = WhisperContext::new(&model_path.to_string_lossy()).expect("Failed to load model");

    let params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });

    let mut state = ctx.create_state().expect("failed to create state");
    state.full(params, &ingested_wav).expect("failed to run model");

    // fetch the results
    let num_segments = state.full_n_segments().expect("failed to get number of segments");

    println!("{}", num_segments);

    for i in 0..num_segments {
        let segment = state.full_get_segment_text(i).expect("failed to get segment");
        let start_timestamp = state.full_get_segment_t0(i).expect("failed to get segment start timestamp");
        let end_timestamp = state.full_get_segment_t1(i).expect("failed to get segment end timestamp");
        println!("[{} - {}]: {}", start_timestamp, end_timestamp, segment);
    }
}
