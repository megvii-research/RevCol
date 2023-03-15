class TrainingHooks():
    def __init__(self, tb_writer):
        self.variance = {}
        self.grad = {}
        self.hooks = []
        self.tb_writer = tb_writer

    def register_hook(self, module, module_prefix):
        def hook_fn_forward( module, input, output):
            self.variance[module._name] = output.var().cpu().detach().numpy()
        def hook_fn_backward(module, grad_in, grad_out):
            self.grad[module._name] = grad_in[0].flatten()[3900]

        for name, block in module.named_children():
            # import pdb; pdb.set_trace()
            block._name =  module_prefix+name
            hook = block.register_forward_hook(hook_fn_forward)
            self.hooks.append(hook)
            hook = block.register_backward_hook(hook_fn_backward)
            self.hooks.append(hook)
    
    def remove_hook(self,):
        for hook in self.hooks:
            hook.remove()

    def log_to_tb(self, step):
        for k, v in self.variance.items():
            self.tb_writer.add_scalar(f'Var_Pretrain/{k}', v, step)
        self.variance={}
        for k, v in self.grad.items():
            self.tb_writer.add_scalar(f'Grad_Pretrain/{k}', v, step)
        self.grad={}