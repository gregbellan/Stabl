import pandas as pd
import numpy as np


datasets = ["Binary LinearSyntheticData",
            "Binary MultidimensionalRipplingHyperShell",
            "Binary ToroidalWaveData",
            "Regression LinearSyntheticData",
            "Regression MultidimensionalRipplingHyperShell",
            "Regression ToroidalWaveData"]

models = ["Lasso",
          "ElasticNet",
          "RandomForest",
          "XGBoost",
          "STABL Lasso",
          "STABL ElasticNet",
          "STABL RandomForest",
          "STABL XGBoost"]

for dataset in datasets:
    print(f"-------------------- Dataset: {dataset} --------------------")
    main_path = f"FINAL Benchmarks results\{dataset}\Training CV\Selected Features "
    
    for model in models:  
        path = f"{main_path}{model}.csv"
        df = pd.read_csv(path)
        actual_informative = np.arange(1, 16) if dataset not in ["Binary ToroidalWaveData", "Regression ToroidalWaveData"] else np.arange(1, 4)
        selected_features = df["Fold selected features"]
        ious = []
        for i in range(len(selected_features)):
            list_features = selected_features[i].replace("[", "").replace("]", "").replace("'", "").split(", ")
            if list_features == ['']:
                ious.append(0)
            else:
                features = np.array([int(x[1:]) for x in list_features])
                iou = len(np.intersect1d(features, actual_informative)) / len(np.union1d(features, actual_informative))
                ious.append(iou)
        iou = np.mean(ious)
        print(f"{model}: {iou:.2f}")