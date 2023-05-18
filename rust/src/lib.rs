use numpy::{IntoPyArray, PyArray2, PyArray1};
use ndarray::Array2;
use ndarray::prelude::*;
use pyo3::prelude::*;
use fanova::{FanovaOptions, RandomForestOptions};

#[pyfunction]
fn fanova_calculate_py(
    py: Python,
    features: &PyArray2<f64>,
    targets: &PyArray1<f64>,
) -> Py<PyArray1<f64>> {
    let features_array: Array2<f64>;
    let targets_vec: Vec<f64>;

    unsafe {
        features_array = features.as_array().to_owned();
        targets_vec = targets.as_array().to_owned().to_vec();
    }

    let features_slices: Vec<Vec<f64>> = features_array
        .axis_iter(Axis(0))
        .map(|row| row.iter().cloned().collect())
        .collect();

    let features_slices_ref: Vec<&[f64]> = features_slices.iter().map(|row| row.as_slice()).collect();

    let mut fanova = FanovaOptions::new()
        .random_forest(RandomForestOptions::new().seed(0))
        .fit(
            features_slices_ref,
            &targets_vec,
        )
        .unwrap();

    let importances: Vec<f64> = (0..features_array.nrows())
        .map(|i| fanova.quantify_importance(&[i]).mean)
        .collect();

    importances.into_pyarray(py).to_owned()
}

#[pymodule]
fn optuna_dashboard_fanova(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fanova_calculate_py, m)?)?;
    Ok(())
}