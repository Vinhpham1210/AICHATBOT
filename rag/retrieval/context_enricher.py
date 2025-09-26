# rag_chatbot/utils/query_processor.py
import json
from typing import List

def enrich_query_with_context(user_query: str, conversation_ctx: List[dict], qwen_api_client) -> str:
    """
    Làm giàu câu hỏi của người dùng bằng ngữ cảnh hội thoại.
    - Luôn luôn viết lại câu hỏi thành câu độc lập, đầy đủ.
    - Thay thế các từ mơ hồ bằng thông tin cụ thể có trong hội thoại.
    - Không thêm thông tin mới ngoài những gì có trong hội thoại.
    """

    print("-> Đang làm giàu câu hỏi bằng ngữ cảnh hội thoại...")

    # Nếu không có lịch sử hội thoại thì trả nguyên câu hỏi gốc
    if not conversation_ctx:
        return user_query

    # Chuyển lịch sử hội thoại sang chuỗi JSON để đưa vào prompt
    history_str = json.dumps(conversation_ctx, ensure_ascii=False, indent=2)

    # Prompt bắt buộc model phải viết lại câu hỏi
    enrichment_prompt = f"""
        Bạn là hệ thống xử lý ngôn ngữ.
        Dựa trên lịch sử hội thoại sau, hãy viết lại câu hỏi hiện tại của người dùng sao cho:
        - Câu hỏi trở thành **độc lập và hoàn chỉnh**, có thể hiểu được mà **không cần nhìn vào lịch sử**.
        - Bạn phải **thay các từ mơ hồ** như "nó", "cái đó", "sản phẩm trên", "các điện thoại trên" bằng **danh sách tên sản phẩm hoặc mô tả cụ thể** có trong hội thoại.
        - Nếu có nhiều sản phẩm, hãy **liệt kê tất cả các tên sản phẩm** đã xuất hiện trong hội thoại vào câu hỏi.
        - Tuyệt đối **không thêm thông tin mới** ngoài những gì đã có trong hội thoại.
        - **Luôn luôn viết lại câu hỏi theo yêu cầu trên (không được giữ nguyên).**

        Lịch sử hội thoại:
        {history_str}

        Câu hỏi hiện tại: "{user_query}"

        Câu hỏi đã làm giàu:
        """


    try: 
        # Gọi LLM với temperature thấp để đảm bảo tính ổn định
        enrichment_response = qwen_api_client.chat.completions.create(
            model="/hdd1/nckh_face/VinhShindo/model",
            messages=[{"role": "user", "content": enrichment_prompt}],
            max_tokens=200,
            temperature=0.1,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
            }
        )
        # Lấy nội dung câu hỏi đã làm giàu
        enriched_query = enrichment_response.choices[0].message.content.strip()

        # Nếu model không trả về gì thì dùng lại câu gốc
        return enriched_query or user_query

    except Exception as e:
        print(f"Lỗi khi làm giàu câu hỏi: {e}. Trả về câu hỏi gốc.")
        return user_query
