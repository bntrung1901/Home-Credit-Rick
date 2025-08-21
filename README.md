🏦 Home Credit Default Risk Prediction
1. Giới thiệu:
- Bối cảnh: Dự đoán rủi ro tín dụng (khả năng vỡ nợ của khách hàng dựa trên dữ liệu tín dụng, hồ sơ cá nhân và lịch sử vay nợ).
- Dữ liệu: Home Credit Default Risk (Kaggle).
- Mục tiêu: Xây dựng mô hình ML dự đoán xác suất khách hàng vỡ nợ, hỗ trợ ngân hàng ra quyết định chấp thuận hoặc từ chối khoản vay.

2. Setup:
- Tạo môi trường ảo (khuyến nghị):
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/MacOS
   venv\Scripts\activate      # Windows
- Cài đặt thư viện: pip install -r requirements.txt

3. Dataset:
- Download the original dataset locally (Kaggle: https://www.kaggle.com/datasets/megancrenshaw/home-credit-default-risk/data).
- Các file chính:
	. application_train.csv / application_test.csv: Thông tin chính về khách hàng (demographic, thu nhập, tài sản, …).
- Các file phụ:
	. bureau.csv / bureau_balance.csv: Lịch sử tín dụng của khách hàng tại các tổ chức khác.
	. previous_application.csv: Thông tin các khoản vay trước đây tại Home Credit.
	. POS_CASH_balance.csv: Lịch sử giao dịch POS/consumer loan.
	. installments_payments.csv: Lịch sử trả nợ (đúng hạn, trễ hạn).
	. credit_card_balance.csv: Hành vi sử dụng thẻ tín dụng.
- Mục tiêu: TARGET (1 = vỡ nợ, 0 = trả nợ đầy đủ).
- Place the CSV files in data/
- Update the path glob to your machine, e.g.:
 	# before (example Windows path)
	path = r"D:\...\*.csv"
	# after (relative to repo)
	path = r"./data/*.csv"
- 

4. Cấu trúc Project:

├── Notebooks/ Home_credit.ipynb
├── Data/
│   ├── application_train.csv
│   ├── application_test.csv
│   ├── bureau.csv
│   ├── bureau_balance.csv
│   ├── previous_application.csv
│   ├── POS_CASH_balance.csv
│   ├── installments_payments.csv
│   └── credit_card_balance.csv
├── Output/
│   └── submission.csv
├── Images/
│   ├── Correlation Heatmap.png
│   ├── Missing Value.png
│   ├── Top Feature.png
│   ├── Outlier/
│   │	├── Credit.png
│   │	├── Income.png
│   ├──	Numerical/
│   │	├── AMT_ANNUITY.png
│   │	├── AMT_CREDIT.png
│   │	├── AMT_INCOME_TOTAL.png
│   │	├── DAYS_BIRTH.png
│   │	├── DAYS_EMPLOYED.png
│   ├──	Categorical/
│   │	├── CODE_GENDER.png 
│   │	├── FLAG_OWN_CAR.png
│   │	├── FLAG_OWN_REALTY.png
│   │	├── NAME_CONTRACT_TYPE.png
├── README.md
├── LICENSE.md
└── requirements.txt

5. Quy trình (Pipeline):
- Khám phá dữ liệu (EDA):
	. Đọc & kiểm tra dữ liệu.
	. Phân phối nhãn (TARGET).
	. Thống kê các biến:
		. Biến Numerical: "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "DAYS_BIRTH", "DAYS_EMPLOYED"
		. Biến Categorical: "NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY"
	. ![Missing values overview](images/Missing Value.png)
	. ![Correlation Heatmap](images/Correlation Heatmap.png)
	. Outlier detection
- Xử lý dữ liệu & Feature Engineering:
	. Tiền xử lý:
		. Gộp train + test để One-Hot Encode (OHE) các biến phân loại.
		. Chuẩn hóa tên cột (loại ký tự đặc biệt).
		. Điền missing values (-999).
	. Feature Engineering từ các bảng phụ:
		. bureau + bureau_balance: tổng hợp theo SK_ID_CURR (mean, max, min, var, count).
		. previous_application: tổng hợp theo SK_ID_CURR.
		. installments_payments: tạo đặc trưng về tỷ lệ trả nợ đúng hạn.
		. POS_CASH_balance, credit_card_balance: tổng hợp hành vi vay/thanh toán.
	. Chuẩn hóa dữ liệu sau khi merge.
- Mô hình (Modeling):
	. Sử dụng LightGBM Classifier, Logistic Regression (chuẩn hóa dữ liệu, regularization).
	. Chiến lược huấn luyện:
		. 5-Fold Cross Validation.
		. Early stopping để tránh overfitting.
	. Siêu tham số (params): learning_rate, num_leaves, max_depth, feature_fraction, …
- Đánh giá mô hình (Evaluation):
	. Cross-validation AUC trung bình trên 5 folds.
	. AUC-ROC
	. Lấy Out-of-Fold predictions.
	. ![Top 30 Feature Importances](images/Top Feature.png)
- Kết quả Submission:
	. File submission.csv gồm: 
		SK_ID_CURR, TARGET  
		100001, 0.0450  
		100005, 0.1174  
		…  

6. Kết quả:
- Baseline Logistic Regression: AUC ~0.65
- LightGBM với Feature Engineering: AUC ~0.77 (Cross-validation)

7. Insight từ Feature Importance: 
- DAYS_BIRTH (tuổi khách hàng): Người trẻ tuổi có rủi ro cao hơn.
- EXT_SOURCE_1/2/3 (điểm tín dụng ngoài): Yếu tố quan trọng nhất, thể hiện sức mạnh của external credit scoring.
- DAYS_EMPLOYED (số ngày đi làm): Người có việc làm ổn định có rủi ro thấp hơn.
- PAYMENT_RATIO (tỷ lệ số tiền đã trả / số tiền phải trả): Trả đúng hạn giảm đáng kể khả năng vỡ nợ.

8. Công nghệ sử dụng:
- Python: pandas, numpy, matplotlib, seaborn
- Machine Learning: scikit-learn, LightGBM
- Visualization: matplotlib, seaborn
- Environment: Jupyter Notebook để phân tích và trình bày kết quả.
- GitHub để lưu trữ và chia sẻ dự án.

9. Hướng phát triển:
- Thử nghiệm thêm các mô hình: XGBoost, CatBoost, Neural Network.
- Feature selection để giảm kích thước dữ liệu.
- Tối ưu hyperparameters bằng Optuna/GridSearch.
- Ensemble nhiều mô hình.
- Triển khai thành API để scoring khách hàng mới.

📄 License
MIT — see LICENSE.

👤 Tác giả
Leo Trung - Email: bntrung1901@gmail.com

