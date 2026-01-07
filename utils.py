def stream_chunks_from_file(file_path, chunk_size=800, overlap=100):
    buffer = []
    buffer_len = 0

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            for word in line.split():
                buffer.append(word)
                buffer_len += 1

                if buffer_len >= chunk_size:
                    yield " ".join(buffer)
                    buffer = buffer[-overlap:]
                    buffer_len = len(buffer)

    if buffer:
        yield " ".join(buffer)
