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

fn generate_silence(duration_ms: u32, output_path: &Path) -> bool {
    if output_path.exists() {
        return true;
    }

    let duration_seconds = (duration_ms as f64) / 1000.0;
    let status = Command::new("ffmpeg")
        .args(&[
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=24000:cl=mono",
            "-t",
            &format!("{duration_seconds:.3}"),
            output_path.to_string_lossy().as_ref(),
        ])
        .status();
    status.map(|s| s.success()).unwrap_or(false)
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

fn write_concat_list(list_path: &Path, paths: &[PathBuf], silence_name: &str) -> PyResult<()> {
    let mut file = File::create(list_path)?;
    for (i, path) in paths.iter().enumerate() {
        let file_name = path
            .file_name()
            .and_then(|s| s.to_str())
            .ok_or_else(|| PyValueError::new_err("音频文件名无效"))?;
        writeln!(file, "file '{}'", file_name)?;
        if i < paths.len() - 1 {
            writeln!(file, "file '{}'", silence_name)?;
        }
    }
    Ok(())
}

#[pyfunction(signature=(input_dir, output_file, pause_ms=400, target_lufs=-16.0, true_peak=-1.5, lra=11.0))]
fn merge_audio_pro(
    input_dir: String,
    output_file: String,
    pause_ms: u32,
    target_lufs: f32,
    true_peak: f32,
    lra: f32,
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

    let silence_file = dir_path.join(format!("silence_{}ms.wav", pause_ms));
    if !generate_silence(pause_ms, &silence_file) {
        return Err(PyRuntimeError::new_err(
            "生成静音片段失败，请确认 ffmpeg 可用",
        ));
    }
    let silence_name = silence_file
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| PyValueError::new_err("静音文件名无效"))?;

    let list_path = dir_path.join("concat_list.txt");
    write_concat_list(&list_path, &paths, silence_name)?;

    let output_path = {
        let output = Path::new(&output_file);
        if output.is_absolute() {
            output.to_path_buf()
        } else {
            std::env::current_dir()?.join(output)
        }
    };

    println!(
        "🚀 Rust 引擎开始混音，共 {} 个片段，停顿 {}ms...",
        paths.len(),
        pause_ms
    );
    let filter = format!("loudnorm=I={}:TP={}:LRA={}", target_lufs, true_peak, lra);
    let status = Command::new("ffmpeg")
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
            "-c:a",
            "libmp3lame",
            "-q:a",
            "2",
            output_path.to_string_lossy().as_ref(),
        ])
        .output()?;

    // 无论成功失败都尝试清理临时文件
    let _ = fs::remove_file(&list_path);

    if status.status.success() {
        println!("✅ 合并成功！输出文件: {}", output_path.display());
        Ok(true)
    } else {
        let err_msg = String::from_utf8_lossy(&status.stderr);
        println!("❌ 合并失败: {}", err_msg);
        Ok(false)
    }
}

#[pyfunction]
fn merge_audio(input_dir: String, output_file: String, pause_ms: u32) -> PyResult<bool> {
    // 兼容旧接口，转调新实现
    merge_audio_pro(input_dir, output_file, pause_ms, -16.0, -1.5, 11.0)
}

/// A Python module implemented in Rust.
#[pymodule(name = "gnosis_rs")]
fn novel_cast_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(clean_text, m)?)?;
    m.add_function(wrap_pyfunction!(merge_audio, m)?)?;
    m.add_function(wrap_pyfunction!(merge_audio_pro, m)?)?;
    Ok(())
}
