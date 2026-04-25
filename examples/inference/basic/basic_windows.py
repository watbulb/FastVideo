"""Single-GPU FastVideo inference example for Windows.

Uses Torch SDPA (no Triton, no Flash-Attn) and disables `pin_cpu_memory`,
which avoids the runtime constraints described in
docs/getting_started/installation/windows.md.
"""

from fastvideo import VideoGenerator

OUTPUT_PATH = "video_samples_windows"


def main() -> None:
    generator = VideoGenerator.from_pretrained(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=False,
    )

    prompt = (
        "A curious raccoon peers through a vibrant field of yellow sunflowers, "
        "its eyes wide with interest. The playful yet serene atmosphere is "
        "complemented by soft natural light filtering through the petals. "
        "Mid-shot, warm and cheerful tones."
    )
    generator.generate_video(prompt, output_path=OUTPUT_PATH, save_video=True)


if __name__ == "__main__":
    main()
