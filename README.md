## Para instalar o python 3.11

```bash
winget install Python.Python.3.11  

# Verifica se instalou corretamente
py -3.11 --version
```


## Criar um venv e entrar

``` bash
python3.11 -m venv venv

# Windows cmd
venv\Scripts\activate.bat

# Windows PowerShell
.\venv\Scripts\Activate.ps1

# Linux e macOS
source venv/bin/activate

# Git Bash no Windows
source venv/Scripts/activate

```
## Instalando bibliotecas

```bash
pip install -r requeriments.txt
```