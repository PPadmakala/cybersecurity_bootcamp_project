#!/usr/bin/env python3
"""Offline NSL-KDD classifier with training and CSV batch prediction."""
import argparse
from pathlib import Path
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')

import io
import csv
from collections import Counter

def validate_upload(raw_bytes, expected_features):
    """Accept only a headered CSV containing the model's 41 feature columns."""
    if len(raw_bytes) > 10 * 1024 * 1024:
        raise ValueError('Upload a CSV smaller than 10 MB.')
    try:
        text = raw_bytes.decode('utf-8-sig')
    except UnicodeDecodeError as exc:
        raise ValueError('Save the CSV using UTF-8 encoding.') from exc
    rows = csv.reader(io.StringIO(text))
    header = next(rows, None)
    if not header:
        raise ValueError('The CSV is empty. Use the provided template.')
    header = [name.strip() for name in header]
    duplicates = [name for name, count in Counter(header).items() if count > 1]
    if duplicates:
        raise ValueError('Duplicate column names: ' + ', '.join(duplicates))
    missing = sorted(set(expected_features) - set(header))
    extra = sorted(set(header) - set(expected_features))
    if missing or extra:
        raise ValueError(f'Columns do not match. Missing: {missing}. Unexpected: {extra}. Use the template; exclude labels and difficulty scores.')
    for line, row in enumerate(rows, start=2):
        if len(row) != len(header):
            raise ValueError(f'CSV line {line} has {len(row)} fields; expected {len(header)}.')
    frame = pd.read_csv(io.StringIO(text))
    frame.columns = header
    if frame.empty:
        raise ValueError('The CSV has no connection records.')
    if len(frame) > 50000:
        raise ValueError('Upload at most 50,000 connections per batch.')
    frame = frame.loc[:, expected_features].copy()
    categories = ['protocol_type', 'service', 'flag']
    for col in categories:
        if frame[col].isna().any():
            raise ValueError(f'Missing values in {col}.')
        frame[col] = frame[col].astype(str).str.strip()
        if frame[col].eq('').any():
            raise ValueError(f'Empty values in {col}.')
    for col in expected_features:
        if col not in categories:
            frame[col] = pd.to_numeric(frame[col], errors='coerce')
            if not np.isfinite(frame[col].to_numpy(dtype=float)).all():
                raise ValueError(f'{col} contains missing, nonnumeric or infinite values.')
    return frame

def predict_batch(frame, fitted_pipeline):
    pred = fitted_pipeline.predict(frame)
    result = frame.copy()
    result.insert(0, 'record_number', np.arange(1, len(frame) + 1))
    result['prediction'] = np.where(pred == 1, 'Attack', 'Normal')
    return result


def train_models(data_dir, output_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    previous_dir = Path.cwd()
    os.chdir(data_dir)
    try:
        import hashlib, json, shutil, time
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import sklearn, joblib
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.pipeline import Pipeline
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.dummy import DummyClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                                     classification_report, confusion_matrix, ConfusionMatrixDisplay)
        OUT = output_dir
        OUT.mkdir(exist_ok=True)
        print('scikit-learn:', sklearn.__version__)
        
        BASE = 'https://raw.githubusercontent.com/defcom17/NSL_KDD/master/'
        def load_data(filename, expected_rows):
            path = Path(filename)
            if not path.exists():
                import urllib.request
                urllib.request.urlretrieve(BASE + filename.replace('+', '%2B'), path)
            frame = pd.read_csv(path, header=None)
            assert frame.shape == (expected_rows, 43), f'Unexpected file shape: {frame.shape}'
            print(filename, frame.shape, 'SHA256:', hashlib.sha256(path.read_bytes()).hexdigest())
            return frame
        train = load_data('KDDTrain+.txt', 125973)
        test = load_data('KDDTest+.txt', 22544)
        features = ['feature_' + str(i) for i in range(41)]
        features[1:4] = ['protocol_type', 'service', 'flag']
        def split_xy(frame):
            X = frame.iloc[:, :41].copy()
            X.columns = features
            labels = frame.iloc[:, 41].astype(str).str.strip().str.rstrip('.')
            y = (labels != 'normal').astype(int)
            assert X.notna().all().all() and labels.notna().all()
            return X, y
        X_train, y_train = split_xy(train)
        X_test, y_test = split_xy(test)
        print(pd.DataFrame({'train': y_train.value_counts(), 'test': y_test.value_counts()})
                .rename(index={0:'Normal', 1:'Attack'}))
        
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)
        cat = ['protocol_type', 'service', 'flag']
        num = [c for c in features if c not in cat]
        def pipeline(estimator):
            prep = ColumnTransformer([
                ('categorical', OneHotEncoder(handle_unknown='ignore'), cat),
                ('numeric', 'passthrough', num)])
            return Pipeline([('preprocess', prep), ('classifier', estimator)])
        models = {
            'Decision Tree': pipeline(DecisionTreeClassifier(max_depth=20, random_state=42)),
            'Random Forest': pipeline(RandomForestClassifier(n_estimators=100, max_depth=20,
                                                            n_jobs=-1, random_state=42))}
        validation = []
        for name, model in models.items():
            start = time.time()
            model.fit(X_fit, y_fit)
            pred = model.predict(X_val)
            validation.append({'model':name, 'validation_f1':f1_score(y_val, pred),
                               'training_seconds':time.time()-start})
        val_table = pd.DataFrame(validation).sort_values('validation_f1', ascending=False)
        selected_name = val_table.iloc[0]['model']
        print(val_table)
        print('Selected by validation F1:', selected_name)
        val_table.to_csv(OUT / 'validation_results.csv', index=False)
        
        rows = []
        for name, model in models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0,1]).ravel()
            rows.append({'model':name, 'accuracy':accuracy_score(y_test,pred),
                         'attack_precision':precision_score(y_test,pred,zero_division=0),
                         'attack_recall':recall_score(y_test,pred,zero_division=0),
                         'attack_f1':f1_score(y_test,pred,zero_division=0),
                         'false_positive_rate':fp/(fp+tn)})
            print(name)
            report = classification_report(y_test,pred,labels=[0,1],target_names=['Normal','Attack'],zero_division=0)
            print(report)
            stem = name.lower().replace(' ', '_')
            (OUT / (stem + '_report.txt')).write_text(report)
            ConfusionMatrixDisplay.from_predictions(y_test,pred,labels=[0,1],display_labels=['Normal','Attack'],cmap='Blues')
            plt.title(name + ' — held-out test data')
            plt.tight_layout()
            plt.savefig(OUT / (stem + '_confusion_matrix.png'), dpi=160)
            plt.close()
        results = pd.DataFrame(rows)
        print(results)
        results.to_csv(OUT / 'test_results.csv',index=False)
        baseline = DummyClassifier(strategy='most_frequent').fit(X_train,y_train)
        print('Majority baseline test accuracy:', accuracy_score(y_test,baseline.predict(X_test)))
        results.set_index('model')[['attack_precision','attack_recall','attack_f1']].plot.bar(ylim=(0,1),rot=0)
        plt.ylabel('Score'); plt.tight_layout()
        plt.savefig(OUT / 'model_comparison.png',dpi=160); plt.close()
        
        selected = models[selected_name]

        # Explain the selected tree-based model using its learned feature importances.
        # The preprocessing step expands categorical values into one-hot encoded columns.
        fitted_preprocessor = selected.named_steps['preprocess']
        fitted_classifier = selected.named_steps['classifier']
        transformed_feature_names = fitted_preprocessor.get_feature_names_out()
        importances = fitted_classifier.feature_importances_
        if len(transformed_feature_names) != len(importances):
            raise ValueError('Feature names and importance values do not match.')
        importance_table = pd.DataFrame({
            'feature': transformed_feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        importance_table.to_csv(OUT / 'feature_importance.csv', index=False)

        top_features = importance_table.head(20).sort_values('importance', ascending=True)
        plt.figure(figsize=(10, 8))
        plt.barh(top_features['feature'], top_features['importance'])
        plt.xlabel('Feature importance')
        plt.ylabel('Network feature')
        plt.title(f'Top 20 Feature Importances — {selected_name}')
        plt.tight_layout()
        plt.savefig(OUT / 'feature_importance_top20.png', dpi=160)
        plt.close()
        print('Top feature importances:')
        print(importance_table.head(10).to_string(index=False))

        X_test.sample(100, random_state=42).to_csv(OUT / 'example_network_connections.csv', index=False)
        sample = X_test.sample(10, random_state=42)
        demo = pd.DataFrame({'actual':y_test.loc[sample.index], 'predicted':selected.predict(sample)})
        demo = demo.replace({0:'Normal',1:'Attack'})
        print(demo)
        demo.to_csv(OUT / 'demo_predictions.csv',index=False)
        joblib.dump(selected, OUT / 'selected_pipeline.joblib')
        (OUT / 'run_metadata.json').write_text(json.dumps({'selected_model':selected_name,
            'selection_rule':'highest validation attack F1', 'seed':42,
            'sklearn_version':sklearn.__version__, 'train_rows':len(train), 'test_rows':len(test)},indent=2))
        shutil.make_archive(str(OUT.parent / (OUT.name + '_archive')), 'zip', OUT)
        print('Results saved to', OUT)
    finally:
        os.chdir(previous_dir)


def run_prediction(csv_path, model_path, output_path):
    # Load only your own trusted model files; joblib loading can execute code.
    model = joblib.load(model_path)
    if not hasattr(model, 'feature_names_in_'):
        raise ValueError('The model must be a fitted pipeline from this project.')
    frame = validate_upload(csv_path.read_bytes(), list(model.feature_names_in_))
    result = predict_batch(frame, model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    counts = result['prediction'].value_counts()
    print('Total connections:', len(result))
    print('Predicted Normal:', int(counts.get('Normal', 0)))
    print('Predicted Attack:', int(counts.get('Attack', 0)))
    print('Predictions are suspected classes, not verified attacks.')
    print('Results saved to', output_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    train = subs.add_parser('train', help='Download NSL-KDD, train models and save results')
    train.add_argument('--data-dir', type=Path, default=Path('data'))
    train.add_argument('--output-dir', type=Path, default=Path('nids_results'))
    predict = subs.add_parser('predict', help='Predict labels for a matching 41-feature CSV')
    predict.add_argument('--csv', type=Path, required=True)
    predict.add_argument('--model', type=Path, default=Path('nids_results/selected_pipeline.joblib'))
    predict.add_argument('--output', type=Path, default=Path('network_connection_predictions.csv'))
    args = parser.parse_args()
    try:
        if args.command == 'train':
            train_models(args.data_dir.resolve(), args.output_dir.resolve())
        else:
            run_prediction(args.csv, args.model, args.output)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
