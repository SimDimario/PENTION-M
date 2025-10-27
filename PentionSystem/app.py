import subprocess

def main():
    comando = ["python", "-m", "streamlit", "run", "ui.py", "--server.port=8501"]
    subprocess.run(comando)

if __name__ == "__main__":
    main()
