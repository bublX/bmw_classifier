import os
import sys

print("=" * 50)
print("🚗 BMW Sales Classification Predictor")
print("=" * 50)
print("\nЗапуск приложения...")
print("👉 Для остановки нажмите Ctrl+C в этом окне")
print("=" * 50)
os.system(f"{sys.executable} -m streamlit run classifier.py")
