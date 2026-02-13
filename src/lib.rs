use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use std::process::Command;

use pyo3::prelude::*;

#[pyfunction]
fn clean_text(text: String) -> String {
    text.lines()
        .map(|line| line.trim())
        .filter(|line| !line.is_empty())
        .collect::<Vec<&str>>()
        .join("\n")
}

fn generate_silence(duration_ms: u32, output_path: &str) -> bool {
    if Path::new(output_path).exists() {
        return true;
    }

    let status = Command::new("ffmpeg")
        .args(&[
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=24000:cl=mono",
            "-t",
            &format!("0.{}", duration_ms), // e.g., "0.500"
            "-acodec",
            "libmp3lame",
            "-ab",
            "48k",
            output_path,
        ])
        .status();
    status.map(|s| s.success()).unwrap_or(false)
}

#[pyfunction]
fn merge_audio(input_dir: String, output_file: String, pause_ms: u32) -> PyResult<bool> {
    let dir_path = Path::new(&input_dir);

    let silence_file = format!("{}/silence_{}ms.mp3", input_dir, pause_ms);
    generate_silence(pause_ms, &silence_file);
    let silence_name = Path::new(&silence_file)
        .file_name()
        .unwrap()
        .to_str()
        .unwrap();

    let mut paths: Vec<_> = fs::read_dir(dir_path)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|p| {
            p.extension().and_then(|s| s.to_str()) == Some("mp3")
                && !p
                    .file_name()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .starts_with("silence_")
        })
        .collect();

    paths.sort();

    let list_path = format!("{}/concat_list.txt", input_dir);
    let mut file = File::create(&list_path)?;

    for (i, path) in paths.iter().enumerate() {
        let file_name = path.file_name().unwrap().to_str().unwrap();
        writeln!(file, "file '{}'", file_name)?;

        // 在两句话之间插入静音，最后一句不插
        if i < paths.len() - 1 {
            writeln!(file, "file '{}'", silence_name)?;
        }
    }

    println!("🚀 Rust 引擎开始合并音频，共 {} 个片段...", paths.len());
    let status = Command::new("ffmpeg")
        .current_dir(dir_path) // 切换到目录，解决路径问题
        .args(&[
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            "concat_list.txt",
            "-af",
            "loudnorm=I=-16:TP=-1.5:LRA=11",
            "-c:a",
            "libmp3lame",
            "-q:a",
            "2",
            &output_file,
        ])
        .output()?; // 使用 output 而不是 status，可以隐藏 ffmpeg 冗长的日志

    if status.status.success() {
        println!("✅ 合并成功！输出文件: {}", output_file);
        // 清理临时文件 (可选)
        let _ = fs::remove_file(list_path);
        Ok(true)
    } else {
        let err_msg = String::from_utf8_lossy(&status.stderr);
        println!("❌ 合并失败: {}", err_msg);
        Ok(false)
    }
}

/// A Python module implemented in Rust.
#[pymodule(name = "gnosis_rs")]
fn novel_cast_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(clean_text, m)?)?;
    m.add_function(wrap_pyfunction!(merge_audio, m)?)?;
    Ok(())
}
