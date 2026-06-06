# HARTH Activity Recognition

Bài 2 sử dụng bộ dữ liệu **HARTH** để nhận dạng hoạt động của con người từ tín hiệu cảm biến gia tốc. Code gồm hai nhánh thực nghiệm:

- Machine Learning trên đặc trưng thống kê trích xuất từ cửa sổ tín hiệu.
- Deep Learning trực tiếp trên chuỗi thời gian.

## Cấu trúc

- `project/`: mã nguồn chính.
- `data/`: dữ liệu CSV sau khi tải về.
- `output/`: kết quả, hình ảnh và bảng tổng hợp.
- `main.py`: file chạy chính.

## Dữ liệu

Link Google Drive chứa dữ liệu:

```text
DATA_URL=https://drive.google.com/drive/folders/1n9zw0lvPfXLTtFR1VOCbd5TmmSHXKUt_?usp=drive_link
```

Nếu thư mục `data/` chưa có CSV, chương trình sẽ tự tải dữ liệu từ link trên về `Source/Task2/data`.

Thông tin dữ liệu:

- Số subject: `22`
- Feature dùng trong code: `back_x`, `back_y`, `back_z`, `thigh_x`, `thigh_y`, `thigh_z`
- Nhãn hoạt động: `1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 130, 140`
- Chia dữ liệu: `16` train subjects, `3` validation subjects, `3` test subjects
- Window size: `128`, step size: `64`

## Cách chạy

Cài thư viện từ thư mục gốc của bài nộp:

```bash
pip install -r requirements.txt
```

Chạy Bài 2:

```bash
cd Source/Task2
python main.py --quick
```

Chạy đầy đủ:

```bash
python main.py
```

## Mô hình

Machine Learning:

- Decision Tree
- Logistic Regression
- KNN
- Linear SVM
- Random Forest
- XGBoost
- SelectKBest + Logistic Regression
- Soft Voting
- Stacking
- Tuned XGBoost

Deep Learning:

- 1D CNN
- 1D CNN + Augmentation
- CNN-LSTM

## Kết quả đầu ra

Các file được lưu trong `output/`:

- `all_results.csv`
- `best_classification_report.csv`
- `eda_analysis.png`
- `feature_importance.png`
- `deep_learning_curves.png`
- `final_comparison.png`
- `best_confusion_matrix.png`
