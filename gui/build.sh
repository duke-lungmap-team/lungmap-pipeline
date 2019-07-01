pyinstaller.exe \
    --additional-hooks-dir="hooks" \
    --add-data "../../../AppData/Local/Programs/Python/Python37/xgboost/*;xgboost/" \
    --add-data "resources/probe_structure_map.json;." \
    -F lungmap_pipeline.py
