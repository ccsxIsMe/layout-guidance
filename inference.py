import torch
from omegaconf import OmegaConf
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, LMSDiscreteScheduler ,EulerDiscreteScheduler
from my_model import unet_2d_condition
import json
from PIL import Image
from utils import compute_ca_loss, Pharse2idx, draw_box, setup_logger
import hydra
import os
from tqdm import tqdm
from utils import load_text_inversion


# 推理函数
# def inference(device, unet, vae, tokenizer, text_encoder, prompt, bboxes, phrases, cfg, logger):
def inference(device, unet, vae, tokenizer, text_encoder, prompt, bboxes, phrases, cfg, logger, height, width):

    logger.info("Inference")  # 记录推理过程
    logger.info(f"Prompt: {prompt}")  # 记录提示词
    logger.info(f"Phrases: {phrases}")  # 记录短语

    # 获取物体位置
    logger.info("Convert Phrases to Object Positions")
    object_positions = Pharse2idx(prompt, phrases)  # 将短语转换为物体位置，从输入的prompt里面获取物体的位置

    # 编码无条件嵌入
    uncond_input = tokenizer(
        [""] * cfg.inference.batch_size, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt"
    )
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]  # 获取无条件嵌入

    # 编码提示词
    input_ids = tokenizer(
        [prompt] * cfg.inference.batch_size,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    cond_embeddings = text_encoder(input_ids.input_ids.to(device))[0]  # 获取条件嵌入
    text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])  # 合并无条件和条件嵌入
    generator = torch.manual_seed(cfg.inference.rand_seed)  # 设置随机种子生成初始潜在噪声

    # 创建噪声调度器
    # noise_scheduler = LMSDiscreteScheduler(beta_start=cfg.noise_schedule.beta_start, beta_end=cfg.noise_schedule.beta_end,
    #                                        beta_schedule=cfg.noise_schedule.beta_schedule, num_train_timesteps=cfg.noise_schedule.num_train_timesteps)
    # 创建噪声调度器
    # noise_scheduler = EulerDiscreteScheduler(beta_start=cfg.noise_schedule.beta_start,
    #                                          beta_end=cfg.noise_schedule.beta_end,
    #                                          beta_schedule=cfg.noise_schedule.beta_schedule,
    #                                          num_train_timesteps=cfg.noise_schedule.num_train_timesteps)
    noise_scheduler = EulerDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        interpolation_type="linear",
        num_train_timesteps=1000,
        prediction_type="epsilon",
        steps_offset=1,
        timestep_spacing="leading",
        trained_betas=None,
        use_karras_sigmas=False
    )

    # 初始化潜在变量
    # latents = torch.randn(
    #     (cfg.inference.batch_size, 4, 64, 64),
    #     generator=generator,
    # ).to(device)
    # latents = torch.randn((cfg.inference.batch_size, 4, 64, 64), generator=generator).to(device)
    latents = torch.randn((cfg.inference.batch_size, 4, height // 8, width // 8), generator=generator).to(device)
    noise_scheduler.set_timesteps(cfg.inference.timesteps)  # 设置时间步长

    latents = latents * noise_scheduler.init_noise_sigma  # 乘以初始噪声标准差

    loss = torch.tensor(10000)  # 初始化损失

    # 推理过程
    for index, t in enumerate(tqdm(noise_scheduler.timesteps)):
        iteration = 0

        while loss.item() / cfg.inference.loss_scale > cfg.inference.loss_threshold and iteration < cfg.inference.max_iter and index < cfg.inference.max_index_step:
            latents = latents.requires_grad_(True)  # 使潜在变量可求导
            latent_model_input = latents
            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)  # 缩放模型输入
            noise_pred, attn_map_integrated_up, attn_map_integrated_mid, attn_map_integrated_down = \
                unet(latent_model_input, t, encoder_hidden_states=cond_embeddings)  # 预测噪声和注意力图

            # 使用引导更新潜在变量
            loss = compute_ca_loss(attn_map_integrated_mid, attn_map_integrated_up, bboxes=bboxes,
                                   object_positions=object_positions) * cfg.inference.loss_scale  # 计算损失

            grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents])[0]  # 计算梯度

            latents = latents - grad_cond * noise_scheduler.sigmas[index] ** 2  # 更新潜在变量
            iteration += 1
            torch.cuda.empty_cache()  # 清空缓存

        with torch.no_grad():
            latent_model_input = torch.cat([latents] * 2)  # 复制潜在变量

            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)  # 缩放模型输入
            noise_pred, attn_map_integrated_up, attn_map_integrated_mid, attn_map_integrated_down = \
                unet(latent_model_input, t, encoder_hidden_states=text_embeddings)  # 预测噪声和注意力图

            noise_pred = noise_pred.sample  # 获取预测噪声

            # 执行引导
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)  # 分割无条件和条件噪声
            noise_pred = noise_pred_uncond + cfg.inference.classifier_free_guidance * (noise_pred_text - noise_pred_uncond)  # 计算引导噪声

            latents = noise_scheduler.step(noise_pred, t, latents).prev_sample  # 更新潜在变量
            torch.cuda.empty_cache()  # 清空缓存

    with torch.no_grad():
        logger.info("Decode Image...")  # 记录解码图像过程
        latents = 1 / 0.18215 * latents  # 缩放潜在变量
        image = vae.decode(latents).sample  # 解码图像
        image = (image / 2 + 0.5).clamp(0, 1)  # 规范化图像
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()  # 转换为numpy数组
        images = (image * 255).round().astype("uint8")  # 转换为uint8类型
        pil_images = [Image.fromarray(image) for image in images]  # 转换为PIL图像
        return pil_images  # 返回生成的图像

# 主函数
@hydra.main(version_base=None, config_path="conf", config_name="base_config")
def main(cfg):
    # 构建并加载模型
    # 打开并读取UNet配置文件
    with open(cfg.general.unet_config) as f:
        unet_config = json.load(f)  # 使用json库加载配置文件内容
    # config_path = './conf/unet/config.json'
    # unet_config = OmegaConf.load(config_path)

    # 根据配置文件创建并加载预训练的UNet模型
    # unet = unet_2d_condition.UNet2DConditionModel(**unet_config).from_pretrained(cfg.general.model_path,subfolder="unet")
    # 加载 UNet2DConditionModel 模型
    # unet = unet_2d_condition.UNet2DConditionModel(**unet_config)
    # unet = unet.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="unet", force_download=True, low_cpu_mem_usage=False, device_map=None, ignore_mismatched_sizes=True, from_tf=False, from_safetensors=True)

    # 加载 UNet2DConditionModel 模型，指定文件类型为 .safetensors
    unet = unet_2d_condition.UNet2DConditionModel(**unet_config)
    unet = unet.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                subfolder="unet",
                                from_safetensors=True,
                                ignore_mismatched_sizes=True,
                                low_cpu_mem_usage=False,
                                device_map=None)

    vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="vae",
                                        force_download=True)
    tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer",
                                              force_download=True)
    text_encoder = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="text_encoder",
                                                 force_download=True)

    # 如果进行真实图像编辑，加载相关模型和参数
    if cfg.general.real_image_editing:
        text_encoder, tokenizer = load_text_inversion(text_encoder, tokenizer, cfg.real_image_editing.placeholder_token, cfg.real_image_editing.text_inversion_path)
        unet.load_state_dict(torch.load(cfg.real_image_editing.dreambooth_path)['unet'])
        text_encoder.load_state_dict(torch.load(cfg.real_image_editing.dreambooth_path)['encoder'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选择设备

    unet.to(device)  # 将模型移动到设备
    text_encoder.to(device)
    vae.to(device)

    # ------------------ 示例输入 ------------------
    examples = {"prompt": "A white hello kitty toy sitting on the floor, playing with a small purple ball near its feet.",  # 提示词
                "phrases": "hello kitty; ball",  # 短语
                "bboxes": [[[0.1, 0.2, 0.5, 0.8]], [[0.75, 0.6, 0.95, 0.8]]],  # 边界框
                'save_path': cfg.general.save_path  # 保存路径
                }

    # ------------------ 真实图像编辑示例输入 ------------------
    if cfg.general.real_image_editing:
        examples = {"prompt": "A {} is standing on grass.".format(cfg.real_image_editing.placeholder_token),  # 提示词
                    "phrases": "{}".format(cfg.real_image_editing.placeholder_token),  # 短语
                    "bboxes": [[[0.4, 0.2, 0.9, 0.9]]],  # 边界框
                    'save_path': cfg.general.save_path  # 保存路径
                    }
    # 准备保存路径
    if not os.path.exists(cfg.general.save_path):
        os.makedirs(cfg.general.save_path)
    logger = setup_logger(cfg.general.save_path, __name__)  # 设置日志记录器

    logger.info(cfg)  # 记录配置信息
    # 保存配置信息
    logger.info("save config to {}".format(os.path.join(cfg.general.save_path, 'config.yaml')))
    OmegaConf.save(cfg, os.path.join(cfg.general.save_path, 'config.yaml'))

    # 设置图像大小
    height = cfg.inference.image_height
    width = cfg.inference.image_width

    # 推理
    # pil_images = inference(device, unet, vae, tokenizer, text_encoder, examples['prompt'], examples['bboxes'], examples['phrases'], cfg, logger)
    pil_images = inference(device, unet, vae, tokenizer, text_encoder, examples['prompt'], examples['bboxes'],
                           examples['phrases'], cfg, logger, height, width)
    # 保存示例图像
    for index, pil_image in enumerate(pil_images):
        image_path = os.path.join(cfg.general.save_path, 'example_{}.png'.format(index))
        logger.info('save example image to {}'.format(image_path))
        draw_box(pil_image, examples['bboxes'], examples['phrases'], image_path)  # 绘制边界框并保存图像

if __name__ == "__main__":
    main()
