import openai
from typing import List, Dict

# 设置 OpenAI API 密钥
openai.api_key = "your-api-key"

def generate_candidate_prompts(conversation_history: List[str], labeled_data: List[Dict[str, str]]) -> List[str]:
    """
    利用 Agent1 基于标注数据生成候选提示词
    """
    # 模拟 prompt，让大模型生成多个候选提示词
    labeled_examples = "\n".join([f"Conversation: {data['conversation']}\nLabel: {data['label']}" for data in labeled_data])
    prompt = (
        f"Based on the labeled examples below, generate candidate prompts to analyze if a user needs special intervention.\n\n"
        f"Labeled Examples:\n{labeled_examples}\n\n"
        f"Conversation History:\n{conversation_history}\n\n"
        f"Generate at least 3 candidate prompts:"
    )
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=500,
        n=1,
        stop=None,
    )
    return response['choices'][0]['text'].strip().split("\n")

def score_prompts(prompts: List[str], conversation_history: List[str]) -> List[Dict[str, float]]:
    """
    利用 Agent2 对生成的候选提示词进行排序打分
    """
    scored_prompts = []
    for prompt in prompts:
        scoring_prompt = (
            f"Rate the following prompt for analyzing if a user needs special intervention based on the conversation history below:\n\n"
            f"Conversation History:\n{conversation_history}\n\n"
            f"Prompt: {prompt}\n\n"
            f"Rate the prompt on a scale of 1 to 10 and provide a brief explanation for the score:"
        )
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=scoring_prompt,
            max_tokens=200,
            n=1,
            stop=None,
        )
        scored_prompts.append({
            "prompt": prompt,
            "score": float(response['choices'][0]['text'].strip().split(" ")[0])  # 假设评分是数字开头
        })
    return sorted(scored_prompts, key=lambda x: x['score'], reverse=True)

def generate_final_data(conversation_history: List[str], final_prompt: str):
    """
    使用最佳提示词生成最终数据
    """
    final_prompt_full = (
        f"Conversation History:\n{conversation_history}\n\n"
        f"Use the following prompt to generate data:\n{final_prompt}\n\n"
        f"Output:"
    )
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=final_prompt_full,
        max_tokens=500,
        n=1,
        stop=None,
    )
    return response['choices'][0]['text'].strip()

# 示例输入
conversation_history = [
    "User: I'm feeling really down today.",
    "Agent: I'm sorry to hear that. Would you like to talk more about it?",
    "User: I don't think anyone can help me."
]
labeled_data = [
    {"conversation": "User: I feel hopeless. Agent: I'm here to listen. User: No one understands.", "label": "Needs intervention"},
    {"conversation": "User: Today was a good day. Agent: That's great to hear! User: Thanks for listening.", "label": "No intervention needed"}
]

# 步骤 1：生成候选提示词
candidate_prompts = generate_candidate_prompts(conversation_history, labeled_data)
print("Candidate Prompts:", candidate_prompts)

# 步骤 2：对候选提示词排序打分
scored_prompts = score_prompts(candidate_prompts, conversation_history)
print("Scored Prompts:", scored_prompts)

# 步骤 3：选出最佳提示词并生成最终数据
best_prompt = scored_prompts[0]['prompt']
final_data = generate_final_data(conversation_history, best_prompt)
print("Final Data:", final_data)
