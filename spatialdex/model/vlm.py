from transformers import PaliGemmaConfig, PaliGemmaForConditionalGeneration, PaliGemmaProcessor
from transformers.modeling_utils import no_init_weights


def build_vlm(vlm_config):
    path = vlm_config["pretrained_model_name_or_path"]
    processor = PaliGemmaProcessor.from_pretrained(path)
    config = PaliGemmaConfig.from_pretrained(path, local_files_only=True)
    with no_init_weights():
        model = PaliGemmaForConditionalGeneration(config)
    return processor, model
