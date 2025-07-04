# 🗺️ IncredibleQuest: Image-Based AI Adventure Generator

> ✨ A creative Python application that transforms a collection of real-world images into an interactive, branching text adventure using AI vision and language models.

## 🧠 Overview

**IncredibleQuest** is a Gen-AI storytelling engine that:
- Analyzes landscape images using advanced AI vision models
- Generates immersive story segments inspired by each image
- Creates thematic connections and branching narrative paths
- Outputs a complete markdown-based interactive storybook

Whether you're using photos from Kerala, the Himalayas, or satellite imagery of India, this tool builds an engaging narrative world that adapts to your visual input.

## 🚀 Features

- 🧠 **Image Understanding**: Uses `Gemma 2` vision model to extract scene descriptions.
- ✍️ **AI Story Generation**: Uses `LLaMA 3.3` to craft narrative passages.
- 🔗 **Story Linking**: Auto-generates multiple endings with interconnected storylines.
- 🎨 **Genre Customization**: Switch between *adventure*, *mystery*, *fantasy*, and *sci-fi* styles.
- 📂 **Markdown Output**: Stories are saved as interactive `.md` files with image previews and navigation.
- 💾 **Caching System**: Saves processed results to speed up future runs.
- ⚙️ **Memory Management**: Processes images in batches with configurable timing to prevent overload.

## 📦 Requirements

- Python 3.6+
- Ollama (v0.1.16+)

Install Python dependencies:
```bash
pip install -r requirements.txt
```

Ensure Ollama is installed and running:
```bash
ollama run gemma2:27b
ollama run llama3.3:latest
```

## 🖼️ Usage

### 🔧 Basic Run
```bash
python incrediblequest.py
```

This will:
- Process all images in the `india_images/` folder  
- Generate stories in `_stories/`  
- Use the default "adventure" narrative style

### ⚙️ Command-Line Options

```bash
python incrediblequest.py   --input india_images   --output _stories   --style fantasy   --length 400   --batch-size 5   --delay 5   --config config/quest_config.json   --no-cache
```

| Option         | Description                                                  |
|----------------|--------------------------------------------------------------|
| `--input`      | Folder containing images (default: `input_images`)           |
| `--output`     | Where to save stories (default: `_stories`)                  |
| `--style`      | "adventure", "mystery", "fantasy", or "sci-fi"               |
| `--length`     | Word count per story segment (default: 300–500)              |
| `--batch-size` | Number of images before long pause (default: 10)             |
| `--delay`      | Time (seconds) between images (default: 5)                   |
| `--config`     | Load settings from a `.json` config file                     |
| `--no-cache`   | Re-run model instead of using cached outputs                 |

## 🔧 Configuration File Example (`config/quest_config.json`)

```json
{
  "input_dir": "india_images",
  "output_dir": "_stories",
  "vision_model": "gemma2:27b",
  "text_model": "llama3.3:latest",
  "story_length": 500,
  "temperature": 0.8,
  "narrative_style": "fantasy",
  "retry_attempts": 3,
  "retry_delay": 2,
  "batch_size": 5,
  "inter_image_delay": 10,
  "start_image": 1,
  "end_image": 20
}
```

## 📂 Output Structure

Once complete, your story folder `_stories/` will include:

- `index.md`:  
  - Auto-generated story title  
  - Story synopsis  
  - Links to start pages  
- `scene_1.md`, `scene_2.md`, etc.:  
  - Image preview  
  - AI-generated passage  
  - Choice links to continue the adventure  

## 🧪 Example Script

Try running:

```bash
chmod +x run_adventure.sh
./run_adventure.sh
```

It will:
- Use sample images
- Set style to “mystery”
- Create a short demo adventure

## 🎨 Customize Your Adventure

You can personalize your journey by:
- 🖼️ Changing the image set (`india_images/`, `my_vacation/`, etc.)
- 🎭 Adjusting narrative tone (e.g., from *epic fantasy* to *ghost mystery*)
- 🔄 Modifying prompts in the code for stylistic variations
- 📤 Adding export options (PDF, HTML, EPUB)

## 💡 Project Ideas

- 🇮🇳 *BharatExplorer*: Use satellite images to simulate a journey across Indian states  
- 🏞 *EcoQuest*: Raise climate awareness by telling stories from endangered landscapes  
- 📸 *MyPhotoAdventure*: Turn personal travel photos into a magical AI journal  

## 🛠 Troubleshooting

See [`TROUBLESHOOTING.md`](./TROUBLESHOOTING.md) for:
- Model loading issues
- Image formatting errors
- Memory overflow handling

## 🙏 Acknowledgments

This project builds upon:
- 🧠 [Ollama](https://ollama.com) for local LLM/vision inference
- 👁️ Gemma 2 Vision Model for image understanding
- ✍️ Llama 3.3 for text generation
