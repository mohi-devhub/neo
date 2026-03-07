from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=6, thread_name_prefix="ollama")
