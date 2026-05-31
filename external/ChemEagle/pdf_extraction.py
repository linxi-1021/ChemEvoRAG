import os
import json
import argparse
from huggingface_hub import hf_hub_download
from pdfmodel.methods import batch_pdf_to_figures_and_tables,_pdf_to_figures_and_tables


def load_config(config_file):
    """Load configurations from JSON file with path adjustments"""
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config['default_image_dir'] = os.path.join(base_dir, config.get('default_image_dir', ''))
    config['default_json_dir'] = os.path.join(base_dir, config.get('default_json_dir', ''))
    config['default_graph_dir'] = os.path.join(base_dir, config.get('default_graph_dir', ''))
    return config


def run_pdf(config_file=None, pdf_dir=None, image_dir=None, model_size= "large"):
    """Main function to run VisualHeist processing"""
    if config_file:
        config = load_config(config_file)
    else:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_config = os.path.join(base_dir, 'scripts/startup.json')
        config = load_config(default_config) if os.path.exists(default_config) else {}
    
    image_dir = image_dir or config.get('image_dir', config.get('default_image_dir'))
    model_size = model_size or config.get('model_size', "base")
    
    print(f"\nProcessing the PDF: {pdf_dir}")
    print(f"Using {'LARGE' if model_size == 'large' else 'BASE'} model")
    _pdf_to_figures_and_tables(pdf_dir, image_dir, large_model=(model_size == "large"))
