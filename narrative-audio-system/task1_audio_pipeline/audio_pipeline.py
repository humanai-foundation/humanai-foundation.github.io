import librosa
import soundfile as sf


def process_audio(input_path, output_path, sample_rate=16000, n_mfcc=13):
    # Resample to a fixed rate so downstream features are comparable across files.
    waveform, loaded_sample_rate = librosa.load(input_path, sr=sample_rate)
    normalized_waveform = librosa.util.normalize(waveform)
    # MFCCs are returned as (coefficients, frames).
    mfcc_features = librosa.feature.mfcc(
        y=normalized_waveform,
        sr=loaded_sample_rate,
        n_mfcc=n_mfcc,
    )
    print("MFCC shape:", mfcc_features.shape)
    sf.write(output_path, normalized_waveform, loaded_sample_rate)
    return mfcc_features


if __name__ == "__main__":
    process_audio("examples/sample_audio.wav", "examples/processed_audio.wav")
