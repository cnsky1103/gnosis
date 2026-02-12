use pyo3::prelude::*;

#[pyfunction]
fn clean_text(text: String) -> String {
    text.lines()
        .map(|line| line.trim())
        .filter(|line| !line.is_empty())
        .collect::<Vec<&str>>()
        .join("\n")
}

/// A Python module implemented in Rust.
#[pymodule(name = "gnosis_rs")]
fn novel_cast_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(clean_text, m)?)?;
    Ok(())
}
