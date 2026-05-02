import modal

app = modal.App("test-connection")

@app.function(memory=2048, timeout=60)
def test():
    import torch, os
    cuda = torch.cuda.is_available()
    return f"Hello from Modal! CUDA: {cuda}, Workspace: OK"

@app.local_entrypoint()
def main():
    result = test.remote()
    print(result)