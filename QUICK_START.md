# 🚀 راهنمای سریع - ETH Forecasting

## شروع فوری (5 دقیقه)

### 1️⃣ نصب وابستگی‌ها
```bash
pip install -r requirements.txt
```

### 2️⃣ مشاهده نتایج به صورت زنده
```bash
python start_server.py
```
سپس مرورگر خود را به آدرس `http://localhost:8000` باز کنید

### 3️⃣ اجرای ارزیابی مدل
```bash
python evaluate_acceptance_rules.py
```

### 4️⃣ بررسی یکپارچگی داده‌ها
```bash
python check_data_leakage.py
```

## 📊 نتایج کلیدی

✅ **دقت جهتی**: 70.0%  
✅ **P-Value**: 0.041 (معنادار آماری)  
✅ **بهبود RMSE**: 34%  
✅ **بهبود MZTAE**: 26%  
✅ **نشت داده**: هیچ موردی شناسایی نشد  

## 📁 فایل‌های مهم

- `reports/ACCEPTANCE_SUMMARY.md` - گزارش کامل ارزیابی
- `reports/data_leakage_report.json` - تحلیل یکپارچگی داده
- `src/` - کد منبع پروژه
- `config/config.yaml` - تنظیمات سیستم

## 🌐 API Endpoints

- `/api/results` - نتایج مدل
- `/api/reports` - لیست گزارش‌ها
- `/api/status` - وضعیت سیستم

## 🔧 تنظیمات

برای تغییر تنظیمات، فایل `config/config.yaml` را ویرایش کنید.

