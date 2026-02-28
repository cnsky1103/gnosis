use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

#[pyfunction]
fn clean_text(text: String) -> String {
    text.lines()
        .map(|line| line.trim())
        .filter(|line| !line.is_empty())
        .collect::<Vec<&str>>()
        .join("\n")
}

fn pause_samples_from_ms(duration_ms: u32, sample_rate: u32) -> u64 {
    let mut pause_samples = ((duration_ms as u64) * (sample_rate as u64) + 500) / 1000;
    if duration_ms > 0 && pause_samples == 0 {
        pause_samples = 1;
    }
    pause_samples
}

fn generate_silence(pause_samples: u64, sample_rate: u32, output_path: &Path) -> bool {
    if pause_samples == 0 {
        return true;
    }

    let data_bytes = match pause_samples.checked_mul(2) {
        Some(value) => value,
        None => return false,
    };
    if data_bytes > u32::MAX as u64 {
        return false;
    }

    let riff_chunk_size = match 36u64.checked_add(data_bytes) {
        Some(value) if value <= u32::MAX as u64 => value as u32,
        _ => return false,
    };
    let data_len_u32 = data_bytes as u32;
    let byte_rate = match sample_rate.checked_mul(2) {
        Some(value) => value,
        None => return false,
    };

    let mut file = match File::create(output_path) {
        Ok(f) => f,
        Err(_) => return false,
    };

    let write_ok = file
        .write_all(b"RIFF")
        .and_then(|_| file.write_all(&riff_chunk_size.to_le_bytes()))
        .and_then(|_| file.write_all(b"WAVE"))
        .and_then(|_| file.write_all(b"fmt "))
        .and_then(|_| file.write_all(&16u32.to_le_bytes()))
        .and_then(|_| file.write_all(&1u16.to_le_bytes()))
        .and_then(|_| file.write_all(&1u16.to_le_bytes()))
        .and_then(|_| file.write_all(&sample_rate.to_le_bytes()))
        .and_then(|_| file.write_all(&byte_rate.to_le_bytes()))
        .and_then(|_| file.write_all(&2u16.to_le_bytes()))
        .and_then(|_| file.write_all(&16u16.to_le_bytes()))
        .and_then(|_| file.write_all(b"data"))
        .and_then(|_| file.write_all(&data_len_u32.to_le_bytes()));
    if write_ok.is_err() {
        return false;
    }

    let mut remaining = data_bytes;
    let chunk = [0u8; 4096];
    while remaining > 0 {
        let write_size = remaining.min(chunk.len() as u64) as usize;
        if file.write_all(&chunk[..write_size]).is_err() {
            return false;
        }
        remaining -= write_size as u64;
    }
    true
}

fn should_include_segment(path: &Path) -> bool {
    let extension = path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase());
    let is_audio = matches!(
        extension.as_deref(),
        Some("wav") | Some("mp3") | Some("m4a") | Some("flac")
    );

    if !is_audio {
        return false;
    }

    if let Some(file_name) = path.file_name().and_then(|s| s.to_str()) {
        return !file_name.starts_with("silence_");
    }
    false
}

fn collect_sorted_segments(input_dir: &Path) -> PyResult<Vec<PathBuf>> {
    let mut paths: Vec<_> = fs::read_dir(input_dir)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|p| should_include_segment(p))
        .collect();
    paths.sort();
    Ok(paths)
}

fn write_concat_list(
    list_path: &Path,
    paths: &[PathBuf],
    silence_name: &str,
    include_silence: bool,
) -> PyResult<()> {
    let mut file = File::create(list_path)?;
    for (i, path) in paths.iter().enumerate() {
        let file_name = path
            .file_name()
            .and_then(|s| s.to_str())
            .ok_or_else(|| PyValueError::new_err("音频文件名无效"))?;
        writeln!(file, "file '{}'", file_name)?;
        if include_silence && i < paths.len() - 1 {
            writeln!(file, "file '{}'", silence_name)?;
        }
    }
    Ok(())
}

#[pyfunction(signature=(input_dir, output_file, pause_ms=400, target_lufs=-16.0, true_peak=-1.5, lra=11.0, silence_sample_rate=24000, keep_master=true))]
fn merge_audio_pro(
    input_dir: String,
    output_file: String,
    pause_ms: u32,
    target_lufs: f32,
    true_peak: f32,
    lra: f32,
    silence_sample_rate: u32,
    keep_master: bool,
) -> PyResult<bool> {
    let dir_path = Path::new(&input_dir);
    if !dir_path.exists() {
        return Err(PyValueError::new_err(format!(
            "输入目录不存在: {}",
            dir_path.display()
        )));
    }

    let paths = collect_sorted_segments(dir_path)?;
    if paths.is_empty() {
        return Err(PyValueError::new_err(format!(
            "未找到可合并的音频片段: {}",
            dir_path.display()
        )));
    }

    if silence_sample_rate == 0 {
        return Err(PyValueError::new_err("silence_sample_rate 必须 > 0"));
    }

    let pause_samples = pause_samples_from_ms(pause_ms, silence_sample_rate);
    let silence_file = dir_path.join(format!(
        "silence_{}ms_{}hz.wav",
        pause_ms, silence_sample_rate
    ));
    if !generate_silence(pause_samples, silence_sample_rate, &silence_file) {
        return Err(PyRuntimeError::new_err(
            "生成静音片段失败，请确认 ffmpeg 可用",
        ));
    }
    let silence_name = silence_file
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| PyValueError::new_err("静音文件名无效"))?;

    let list_path = dir_path.join("concat_list.txt");
    write_concat_list(&list_path, &paths, silence_name, pause_samples > 0)?;

    let output_path = {
        let output = Path::new(&output_file);
        if output.is_absolute() {
            output.to_path_buf()
        } else {
            std::env::current_dir()?.join(output)
        }
    };

    println!(
        "🚀 Rust 引擎开始混音，共 {} 个片段，停顿 {}ms ({} samples @ {}Hz)...",
        paths.len(),
        pause_ms,
        pause_samples,
        silence_sample_rate
    );
    let filter = format!("loudnorm=I={}:TP={}:LRA={}", target_lufs, true_peak, lra);
    let sample_rate_text = silence_sample_rate.to_string();
    let master_path = output_path.with_extension("master.wav");
    let master_status = Command::new("ffmpeg")
        .current_dir(dir_path)
        .args(&[
            "-y",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            "concat_list.txt",
            "-af",
            &filter,
            "-ar",
            &sample_rate_text,
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            master_path.to_string_lossy().as_ref(),
        ])
        .output()?;

    // 无论成功失败都尝试清理临时文件
    let _ = fs::remove_file(&list_path);

    if !master_status.status.success() {
        let err_msg = String::from_utf8_lossy(&master_status.stderr);
        println!("❌ 主母带生成失败: {}", err_msg);
        return Ok(false);
    }

    let output_extension = output_path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_else(String::new);
    let final_status = if output_extension == "wav" {
        if output_path != master_path {
            fs::copy(&master_path, &output_path)?;
        }
        None
    } else if output_extension == "mp3" {
        Some(
            Command::new("ffmpeg")
                .args(&[
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    master_path.to_string_lossy().as_ref(),
                    "-c:a",
                    "libmp3lame",
                    "-q:a",
                    "2",
                    output_path.to_string_lossy().as_ref(),
                ])
                .output()?,
        )
    } else {
        Some(
            Command::new("ffmpeg")
                .args(&[
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    master_path.to_string_lossy().as_ref(),
                    output_path.to_string_lossy().as_ref(),
                ])
                .output()?,
        )
    };

    if let Some(status) = final_status {
        if !status.status.success() {
            let err_msg = String::from_utf8_lossy(&status.stderr);
            println!("❌ 导出失败: {}", err_msg);
            return Ok(false);
        }
    }

    if !keep_master {
        let _ = fs::remove_file(&master_path);
    }

    println!(
        "✅ 合并成功！输出文件: {}（母带: {}）",
        output_path.display(),
        master_path.display()
    );
    Ok(true)
}

#[pyfunction]
fn merge_audio(input_dir: String, output_file: String, pause_ms: u32) -> PyResult<bool> {
    // 兼容旧接口，转调新实现
    merge_audio_pro(
        input_dir,
        output_file,
        pause_ms,
        -16.0,
        -1.5,
        11.0,
        24000,
        true,
    )
}

/// A Python module implemented in Rust.
#[pymodule(name = "gnosis_rs")]
fn novel_cast_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(clean_text, m)?)?;
    m.add_function(wrap_pyfunction!(merge_audio, m)?)?;
    m.add_function(wrap_pyfunction!(merge_audio_pro, m)?)?;
    Ok(())
}
