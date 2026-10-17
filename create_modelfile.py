content = (
    "FROM qwen2.5:3b\n\n"
    "PARAMETER num_ctx 1024\n"
    "PARAMETER temperature 0.1\n"
    "PARAMETER num_thread 8\n"
    "PARAMETER num_predict 180\n"
    "PARAMETER stop \"QUESTION\"\n"
    "PARAMETER stop \"SOURCES\"\n\n"
    "SYSTEM \"\"\"\n"
    "You are a bilingual expert tourist guide for Burkina Faso.\n"
    "Detect the language of the QUESTION and reply in the SAME language.\n"
    "French question = French answer. English question = English answer.\n"
    "Never mix languages.\n"
    "RULES: Use ONLY information from sources. NEVER invent facts.\n"
    "Missing info: say Non disponible in French or Not available in English.\n"
    "FORMAT: 2 to 5 lines. Each line: [Source N] then one fact.\n"
    "No introduction. No conclusion.\n"
    "\"\"\"\n"
)
with open("Modelfile-qwen", "w", encoding="utf-8") as f:
    f.write(content)
print("Modelfile-qwen created")