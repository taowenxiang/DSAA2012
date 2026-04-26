import os
import json
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class StoryReranker:
    def __init__(self, clip_model_path="openai/clip-vit-base-patch32"):
        """
        初始化“赛博裁判” CLIP 模型。
        考虑到你要跑批量测试，把模型加载到 GPU 会快很多。
        """
        print("正在加载 CLIP 模型...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(clip_model_path).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(clip_model_path)
        
        # 权重参数：你可以自己调参，决定是“紧扣提示词”更重要，还是“画面连贯”更重要
        self.alpha_text = 0.5 
        self.beta_image = 0.5

    def get_clip_features(self, image=None, text=None):
        """核心算分工具：将图像或文本转化为向量"""
        inputs = self.processor(text=text, images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs

    def score_image_text(self, image_path, prompt):
        """计算单张图和它对应文本的匹配度"""
        image = Image.open(image_path).convert("RGB")
        outputs = self.get_clip_features(image=image, text=[prompt])
        # 提取相似度分数 (logits_per_image)
        score = outputs.logits_per_image[0][0].item()
        return score

    def score_image_image(self, img_path_1, img_path_2):
        """计算两张图在视觉上的连贯性（当前帧 vs 上一帧）"""
        img1 = Image.open(img_path_1).convert("RGB")
        img2 = Image.open(img_path_2).convert("RGB")
        
        # 获取两张图的特征向量
        inputs1 = self.processor(images=img1, return_tensors="pt").to(self.device)
        inputs2 = self.processor(images=img2, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            feat1 = self.model.get_image_features(**inputs1)
            feat2 = self.model.get_image_features(**inputs2)
        
        # 计算余弦相似度 (Cosine Similarity)
        cos_sim = torch.nn.functional.cosine_similarity(feat1, feat2)
        return cos_sim.item()

    def select_best_sequence(self, panels_data):
        """
        核心逻辑：从一堆候选图里挑出最好的一条线。
        这里我们先用最基础的【贪心算法】（走一步看一步）。
        """
        selected_sequence = []
        prev_image_path = None
        
        # 遍历每一格 (Panel)
        for panel_idx, panel_info in enumerate(panels_data):
            prompt = panel_info["prompt"]
            candidates = panel_info["candidate_images"] # 成员B生成的 K 张图的路径列表
            
            best_score = -float('inf')
            best_img = None
            
            # 评估当前格子的每一张候选图
            for img_path in candidates:
                # 1. 算图文分数
                t_score = self.score_image_text(img_path, prompt)
                
                # 2. 算图像连贯分数 (如果是第一格，就没有上一帧，分数为0)
                i_score = 0
                if prev_image_path is not None:
                    i_score = self.score_image_image(img_path, prev_image_path)
                
                # 3. 综合打分
                total_score = (self.alpha_text * t_score) + (self.beta_image * i_score)
                
                # 记录最高分的图
                if total_score > best_score:
                    best_score = total_score
                    best_img = img_path
            
            # 把这格选出来的图放进最终序列
            selected_sequence.append(best_img)
            # 更新 prev_image_path，供下一格对比用
            prev_image_path = best_img 
            
        return selected_sequence

# --- 使用示例 (伪代码) ---
# 假设你读取了成员A和B生成的中间 JSON 文件
# mock_data = [
#     {"panel": 1, "prompt": "a panda eating an apple...", "candidate_images": ["p1_c1.jpg", "p1_c2.jpg"]},
#     {"panel": 2, "prompt": "the panda sleeping...", "candidate_images": ["p2_c1.jpg", "p2_c2.jpg"]}
# ]
# reranker = StoryReranker()
# final_story = reranker.select_best_sequence(mock_data)
# print("最终选出的图片序列:", final_story)

def main():
    reranker = StoryReranker()
    with open("outputs/intermediate/parsed/01.parsed.json", "r") as f:
        data = json.load(f)
    final_story = reranker.select_best_sequence(data["panels"])
    print(final_story)

if __name__ == "__main__":
    main()
