import re

with open('solution.py', 'r') as f:
    content = f.read()

# 1. Add import for PolyFeatureExpander
old_import = "from models.elasticnet_model import ElasticNetModel"
new_import = """from models.elasticnet_model import ElasticNetModel
from models.poly_features import PolyFeatureExpander"""
content = content.replace(old_import, new_import)

# 2. Add a helper function to apply poly features
poly_helper = """

def _apply_poly_features(xdf, poly_expander):
    if poly_expander is None:
        return xdf
    return poly_expander.transform(xdf)
"""

# 3. Modify create_base_model and train loop
# Add poly expander creation in train_and_evaluate
old_train_func = """def train_and_evaluate(h5dir: Optional[str] = None) -> Dict[str, float]:
    h5dir = _resolve_h5dir(h5dir)
    train_dates, test_dates = train_test_dates()
    feat_gen = MeowFeatureGenerator(cacheDir=None)
    model = _create_base_model()
    model.reset()"""

new_train_func = """def train_and_evaluate(h5dir: Optional[str] = None) -> Dict[str, float]:
    h5dir = _resolve_h5dir(h5dir)
    train_dates, test_dates = train_test_dates()
    feat_gen = MeowFeatureGenerator(cacheDir=None)
    poly_expander = PolyFeatureExpander()
    model = _create_base_model()
    model.reset()"""

content = content.replace(old_train_func, new_train_func)
content = content.replace(
    "def _create_base_model():",
    poly_helper + "def _create_base_model():"
)

# 4. In the training loop, apply poly features after genFeatures
# First training loop (base model)
content = content.replace(
    """        model.partial_fit(xdf, ydf)
        del xdf, ydf
    model.finalize_fit()""",
    """        xdf_poly = _apply_poly_features(xdf, poly_expander)
        model.partial_fit(xdf_poly, ydf)
        del xdf, xdf_poly, ydf
    model.finalize_fit()"""
)

# The second occurrence (interval residual training)
# Need to be more specific
# Find: interval residual training loop
old_ir_loop = """    interval_residual = IntervalResidualRidge()
    for chunk in _chunk_dates(train_dates, N_CHUNKS):
        raw = pd.concat(list(iter_days(h5dir, chunk)), ignore_index=True)
        xdf, ydf = feat_gen.genFeatures(raw)
        del raw
        base_pred = _postprocess_forecast(ydf, model.predict(xdf), forecast_cs_mean_shrink)
        resid = ydf["fret12"].to_numpy(dtype=np.float64, copy=False) - base_pred
        interval_residual.partial_fit(xdf, resid, base_pred=base_pred)
        del xdf, ydf, base_pred, resid"""

new_ir_loop = """    interval_residual = IntervalResidualRidge()
    for chunk in _chunk_dates(train_dates, N_CHUNKS):
        raw = pd.concat(list(iter_days(h5dir, chunk)), ignore_index=True)
        xdf, ydf = feat_gen.genFeatures(raw)
        del raw
        xdf_poly = _apply_poly_features(xdf, poly_expander)
        base_pred = _postprocess_forecast(ydf, model.predict(xdf_poly), forecast_cs_mean_shrink)
        resid = ydf["fret12"].to_numpy(dtype=np.float64, copy=False) - base_pred
        interval_residual.partial_fit(xdf_poly, resid, base_pred=base_pred)
        del xdf, xdf_poly, ydf, base_pred, resid"""

content = content.replace(old_ir_loop, new_ir_loop)

# 5. In the test loop, apply poly features
old_test_loop = """    for chunk in _chunk_dates(test_dates, N_CHUNKS):
        raw = pd.concat(list(iter_days(h5dir, chunk)), ignore_index=True)
        xdf, ydf = feat_gen.genFeatures(raw)
        del raw
        ydf = ydf.copy()
        forecast = _postprocess_forecast(ydf, model.predict(xdf), forecast_cs_mean_shrink)
        forecast = forecast + interval_residual.predict(xdf, base_pred=forecast)
        ydf.loc[:, "forecast"] = forecast
        del xdf"""

new_test_loop = """    for chunk in _chunk_dates(test_dates, N_CHUNKS):
        raw = pd.concat(list(iter_days(h5dir, chunk)), ignore_index=True)
        xdf, ydf = feat_gen.genFeatures(raw)
        del raw
        ydf = ydf.copy()
        xdf_poly = _apply_poly_features(xdf, poly_expander)
        forecast = _postprocess_forecast(ydf, model.predict(xdf_poly), forecast_cs_mean_shrink)
        forecast = forecast + interval_residual.predict(xdf_poly, base_pred=forecast)
        ydf.loc[:, "forecast"] = forecast
        del xdf, xdf_poly"""

content = content.replace(old_test_loop, new_test_loop)

with open('solution.py', 'w') as f:
    f.write(content)

print("Patched successfully")
