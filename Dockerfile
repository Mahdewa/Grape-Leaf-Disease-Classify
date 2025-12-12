# Gunakan Python 3.9
FROM python:3.9

# Buat user baru (Hugging Face mewajibkan user non-root dengan ID 1000)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Set folder kerja
WORKDIR /app

# Copy requirements & Install
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy semua file lain (app.py, model.h5)
COPY --chown=user . .

# Buka port 7860 (Hugging Face wajib pakai port 7860, bukan 5000)
EXPOSE 7860

# Jalankan Flask pakai Gunicorn di port 7860
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]