### LLM desde cero (CLASS_2)

Este módulo contiene una implementación didáctica de modelos de lenguaje (LLM) desde cero para enseñar fundamentos:

- **Tokenización a nivel de caracteres** (vocabulario mínimo, sin dependencias externas)
- **Modelo Bigram** (aprende solo la transición de un carácter al siguiente)
- **Mini-Transformer (tipo GPT)** con self-attention enmascarada, capas residuales y MLP
- Scripts para **entrenar** y **generar** texto

#### 1) Requisitos

1) Instala PyTorch según tu entorno (Mac Intel/Apple Silicon, CUDA/CPU):

```bash
# Recomendado: sigue la guía oficial de PyTorch para tu sistema
# https://pytorch.org/get-started/locally/
```

Ejemplos:

- macOS Apple Silicon (MPS):

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

- macOS Intel (CPU):

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

- Linux/Windows con CUDA 12.1 (si tienes GPU NVIDIA):

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

2) Instala el resto de dependencias:

```bash
python -m pip install -r CLASS_2/requirements.txt
```

#### 2) Datos

Incluimos un dataset pequeño en `CLASS_2/data/tiny_spanish.txt`. Puedes reemplazarlo por cualquier texto propio en español y re-entrenar.

#### 3) Entrenamiento

- Modelo Bigram (rápido y simple):

```bash
python CLASS_2/train.py --model bigram --data CLASS_2/data/tiny_spanish.txt --max-steps 2000
```

- Mini GPT (más capaz, más lento):

```bash
python CLASS_2/train.py --model gpt --data CLASS_2/data/tiny_spanish.txt --max-steps 2000 --n-embd 128 --n-head 4 --n-layer 2
```

Parámetros útiles (comunes):

- `--batch-size` (por defecto 32)
- `--block-size` contexto en tokens (por defecto 128)
- `--lr` tasa de aprendizaje (por defecto 3e-4)
- `--eval-interval` cada cuántos pasos evaluar (por defecto 200)
- `--device` `cpu`/`mps`/`cuda` (detecta automáticamente)

Se guardarán checkpoints en `CLASS_2/checkpoints/` con metadatos necesarios para generar.

#### 4) Generación de texto

Tras entrenar, genera texto con un checkpoint:

```bash
python CLASS_2/generate.py --ckpt CLASS_2/checkpoints/<tu_checkpoint>.pt --prompt "Hola" --max-new-tokens 300 --temperature 0.8 --top-k 50
```

Si no pasas `--prompt`, empieza desde un token de nueva línea.

#### 5) Estructura

- `CLASS_2/data/tiny_spanish.txt`: dataset pequeño de ejemplo
- `CLASS_2/tokenizer.py`: tokenizador de caracteres, encode/decode y persistencia
- `CLASS_2/bigram_model.py`: modelo Bigram en PyTorch
- `CLASS_2/transformer_model.py`: mini GPT desde cero (self-attention, MLP, residual)
- `CLASS_2/train.py`: script de entrenamiento genérico (elige `--model`)
- `CLASS_2/generate.py`: script para generar texto desde un checkpoint
- `CLASS_2/config.py`: configuración y valores por defecto

#### 6) Consejos didácticos

- Empieza con el Bigram para mostrar límites: aprende estadísticas locales, no estructura de largo alcance.
- Luego compara con el Mini-GPT: nota cómo mejora la coherencia al aumentar `--block-size`, capas y cabezas.
- Ajusta `temperature` y `top-k` en generación para mostrar su efecto en creatividad vs. coherencia.

#### 7) MPS (Mac) o CUDA (NVIDIA)

El script detecta automáticamente `mps`/`cuda` si están disponibles. Puedes forzar con `--device cpu` si quieres comparativas.

