import torch.optim as optim


class Build_Optimizer:
    def __init__(self, config, net):
        self.name = "optimizer"
        self.build(config, net)

    def build(self, config, net):

        self.optimizer = None

        if config.OPTIMIZER.NAME == "Adam":
            model_params_group = [
                {
                    "params": net.parameters(),
                    "lr": config.OPTIMIZER.LEARNING_RATE,
                    "weight_decay": 5e-4,
                    "momentum": 0.9,
                }
            ]
            self.optimizer = optim.Adam(model_params_group)

        if config.OPTIMIZER.NAME == "SGD":

            special_modules = [
                net.backbone_classifier,
                net.modal_interaction_classifier,
            ]

            # Ignored parameters
            ignored_params = []
            for module in special_modules:
                ignored_params += list(map(id, module.parameters()))

            # Base parameters
            base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

            # Param groups
            param_groups = [{"params": base_params, "lr": 0.1 * config.OPTIMIZER.LEARNING_RATE}]
            for module in special_modules:
                param_groups.append({"params": module.parameters(), "lr": config.OPTIMIZER.LEARNING_RATE})

            # Optimizer
            self.optimizer = optim.SGD(param_groups, weight_decay=5e-4, momentum=0.9, nesterov=True)
