#%%
from sklearn import metrics
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision.ops import MLP
import torch.nn.functional as F
from dataclasses import dataclass


class DINOv1BaseModel(nn.Module):
    def __init__(self, in_chans=3, model_name="facebook/dino-vits8"):
        super().__init__()
        from transformers import AutoModel #type: ignore
        self.model = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        if in_chans not in [1, 3]:
            raise ValueError(f"Only in_chans=1 or 3 is supported, got {in_chans}")

        if in_chans == 1:
            old_conv = self.model.embeddings.patch_embeddings.projection
            new_conv = nn.Conv2d(
                in_channels=1,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=(old_conv.bias is not None)
            )
            with torch.no_grad():
                new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
                if old_conv.bias is not None:
                    new_conv.bias.data = old_conv.bias.data
            self.model.embeddings.patch_embeddings.projection = new_conv
            self.model.embeddings.patch_embeddings.num_channels = 1
            self.model.config.num_channels = 1

    def forward(self, x, **kwargs):
        return self.model(pixel_values=x, interpolate_pos_encoding=True, **kwargs).last_hidden_state[:, 0]
    
class ViTS8PretrainedModel(nn.Module):
    """
    Wrapper for pretrained DINO ViT-S/8.
    It merges the pretrained RGB patch projection into a one-channel projection when requested.
    """
    def __init__(self, in_chans=1):
        super().__init__()
        self.model = torch.hub.load("facebookresearch/dino:main", "dino_vits8")

        if in_chans not in [1, 3]:
            raise ValueError(f"Only in_chans=1 or 3 are supported, got {in_chans}")

        if in_chans == 1:
            self._replace_patch_projection_with_one_channel_projection()

    def _replace_patch_projection_with_one_channel_projection(self):
        old_proj = self.model.patch_embed.proj
        new_proj = nn.Conv2d(
            in_channels=1,
            out_channels=old_proj.out_channels,
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
            padding=old_proj.padding,
            dilation=old_proj.dilation,
            groups=old_proj.groups,
            bias=old_proj.bias is not None,
            padding_mode=old_proj.padding_mode,
        )

        with torch.no_grad():
            new_proj.weight.copy_(old_proj.weight.sum(dim=1, keepdim=True))
            if old_proj.bias is not None:
                new_proj.bias.copy_(old_proj.bias)

        self.model.patch_embed.proj = new_proj

    def forward(self, x, **kwargs):
        return self.model(x, **kwargs)

class ConvNeXtDINOv3BaseModel(nn.Module):
    """
    Wrapper for DINOv3 ConvNeXt backbone (facebook/dinov3-convnext-*).
    It adapts the first convolutional layer for grayscale input if needed
    and returns global pooled features from the last stage.
    """
    def __init__(self, in_chans=3, model_name="facebook/dinov3-convnext-small-pretrain-lvd1689m"):
        super().__init__()
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(model_name)

        if in_chans not in [1, 3]:
            raise ValueError(f"Only in_chans=1 or 3 are supported, got {in_chans}")

        first_conv_name, first_conv = self._find_first_conv2d(self.model)
        if first_conv.in_channels != in_chans:
            new_conv = nn.Conv2d(
                in_channels=in_chans,
                out_channels=first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                dilation=first_conv.dilation,
                groups=first_conv.groups,
                bias=first_conv.bias is not None,
                padding_mode=first_conv.padding_mode,
            )

            with torch.no_grad():
                if in_chans == 1 and first_conv.in_channels == 3:
                    new_conv.weight.copy_(first_conv.weight.mean(dim=1, keepdim=True))
                else:
                    new_conv.weight.zero_()
                    channels_to_copy = min(in_chans, first_conv.in_channels)
                    new_conv.weight[:, :channels_to_copy].copy_(first_conv.weight[:, :channels_to_copy])
                if first_conv.bias is not None:
                    new_conv.bias.copy_(first_conv.bias)

            self._set_module_by_name(self.model, first_conv_name, new_conv)

        self.model.config.num_channels = in_chans

    @staticmethod
    def _find_first_conv2d(model):
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                return name, module
        raise ValueError("Could not find a Conv2d stem in the DINOv3 ConvNeXt model")

    @staticmethod
    def _set_module_by_name(model, module_name, new_module):
        parent = model
        path = module_name.split(".")
        for part in path[:-1]:
            parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
        last = path[-1]
        if last.isdigit():
            parent[int(last)] = new_module
        else:
            setattr(parent, last, new_module)

    def forward(self, x, **kwargs):
        outputs = self.model(pixel_values=x, **kwargs)
        if getattr(outputs, "pooler_output", None) is not None:
            return outputs.pooler_output
        if getattr(outputs, "last_hidden_state", None) is not None:
            last_hidden_state = outputs.last_hidden_state
            if last_hidden_state.dim() == 4:
                return last_hidden_state.mean(dim=(-2, -1))
            if last_hidden_state.dim() == 3:
                return last_hidden_state[:, 0]
            return last_hidden_state
        raise ValueError("DINOv3 ConvNeXt output has neither pooler_output nor last_hidden_state")
    
def compute_wga_torch(y_true: torch.Tensor, y_pred: torch.Tensor, sensitive: torch.Tensor) -> torch.Tensor:
    worst_group_acc = torch.tensor(1.0, device=y_true.device)
    for label in torch.unique(y_true):
        for bias in torch.unique(sensitive):
            mask = (y_true == label) & (sensitive == bias)
            if mask.any():
                group_acc = (y_pred[mask] == y_true[mask]).float().mean()
                worst_group_acc = torch.minimum(worst_group_acc, group_acc)
    return worst_group_acc

def _safe_rate_torch(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    return num.float() / den.float() if int(den.item()) > 0 else torch.tensor(0.0, device=num.device)

def compute_equalized_odds_torch(y_true: torch.Tensor, y_pred: torch.Tensor, sensitive: torch.Tensor) -> torch.Tensor:
    groups = torch.sort(torch.unique(sensitive)).values
    classes = torch.sort(torch.unique(torch.cat([y_true, y_pred]))).values
    if groups.numel() < 2:
        return torch.tensor(0.0, device=y_true.device)

    gaps = []
    for true_class in classes:
        true_mask = y_true == true_class
        for pred_class in classes:
            rates = []
            valid_groups = []
            for group in groups:
                group_true_mask = true_mask & (sensitive == group)
                den = group_true_mask.sum()
                if int(den.item()) == 0:
                    continue
                rates.append(_safe_rate_torch(((y_pred == pred_class) & group_true_mask).sum(), den))
                valid_groups.append(group)

            for i in range(len(valid_groups)):
                for j in range(i + 1, len(valid_groups)):
                    gaps.append(torch.abs(rates[i] - rates[j]))

    if not gaps:
        return torch.tensor(0.0, device=y_true.device)

    return torch.stack(gaps).mean()

class PretrainedWithLinearProbing(pl.LightningModule):
    def __init__(self,  params):
        super().__init__()  
        self.save_hyperparameters(params.__dict__)  
        self.params = params        

        self.name_data = str(getattr(self.params, 'name_data', 'colorMNIST'))
        self.online_linear_probing = getattr(self.params, 'online_linear_probing', False)
        self.loss_name = getattr(self.params, 'loss_name', 'uow')
        self.use_cross_entropy = self.loss_name in {'crossentropy', 'ce'}

        if self.name_data == 'colorMNIST':
            self.nb_class_linear_probing = 10
            self.nb_bias = 2
            from src.convnet import ConvNet
            self.f = ConvNet(c_in=3, image_width=32)
            in_dim = 128 ; proj_hidden_dim = 512; num_layers=2; out_dim=params.proj_out_dim

        elif self.name_data == 'MNIST':
            self.nb_class_linear_probing = 10
            self.nb_bias = 1
            from src.convnet import ConvNet
            self.f = ConvNet(c_in=1, image_width=32)
            in_dim = 128 ; proj_hidden_dim = 512; num_layers=2; out_dim=params.proj_out_dim

        elif self.name_data in {'celeba', 'celeba_intersectional'}:
            if self.name_data == 'celeba_intersectional':
                self.nb_class_linear_probing = 2
                self.nb_bias = 4
            else:
                experiment = getattr(self.params, 'experiment', None)
                if experiment is None:
                    target_spec = "a"
                    bias_spec = "m"
                else:
                    from dataloaders.celeba import parse_celeba_experiment
                    target_spec, bias_spec = parse_celeba_experiment(experiment)

                from dataloaders.celeba import celeba_num_classes_from_spec
                self.nb_class_linear_probing = celeba_num_classes_from_spec(target_spec)
                self.nb_bias = celeba_num_classes_from_spec(bias_spec)

            import torchvision
            weights = (
                torchvision.models.ResNet18_Weights.DEFAULT
                if getattr(self.params, 'pretrained_backbone', True)
                else None
            )
            resnet = torchvision.models.resnet18(weights=weights)
            feature_dim = list(resnet.children())[-1].in_features 
            resnet.fc = nn.Linear(feature_dim, 256) # type: ignore 

            in_dim = 256 ; proj_hidden_dim = 1024; num_layers=2; out_dim=self.params.proj_out_dim
            self.f = resnet #nn.Sequential(*list(resnet.children())[:-1], nn.Flatten(start_dim=1))

        elif self.name_data in {'birds', 'fairface'}:
            # Waterbirds: bird type (2 classes) with background as bias.
            # FairFace: race (7 classes) with gender as bias.
            if self.name_data == 'birds':
                self.nb_class_linear_probing = 2
                self.nb_bias = 2
            else:
                self.nb_class_linear_probing = 7
                self.nb_bias = 2

            import torchvision
            weights = torchvision.models.ResNet18_Weights.DEFAULT
            resnet = torchvision.models.resnet18(weights=weights)
            feature_dim = list(resnet.children())[-1].in_features
            resnet.fc = nn.Linear(feature_dim, 256) # type: ignore
            in_dim = 256 ; proj_hidden_dim = 1024; num_layers=2; out_dim=self.params.proj_out_dim
            self.f = resnet

        else:
            raise ValueError(f"Unknown dataset {self.name_data}")

        if self.use_cross_entropy:
            self.g = nn.Linear(in_dim, self.nb_class_linear_probing) # type: ignore
        else:
            hidden_channels = [proj_hidden_dim] * (num_layers - 1) + [out_dim]
            self.g = MLP(in_channels=in_dim, hidden_channels=hidden_channels, activation_layer=nn.GELU, norm_layer=nn.BatchNorm1d, dropout=0.0)

        if self.online_linear_probing and not self.use_cross_entropy:
            self.linear_prob_labels = nn.Linear(in_dim, self.nb_class_linear_probing)  # type: ignore
        else:
            self.linear_prob_labels = None

        self.val_preds_labels = []
        self.val_y = []
        self.val_bias = []

        if self.loss_name == 'uow_intersectional':
            from intersectional_uow import IntersectionalUOW
            self.loss = IntersectionalUOW(self.params)
        elif self.loss_name == 'uow':
            from uow import UOW
            self.loss = UOW(self.params)
        elif self.loss_name == 'fscl+':
            from uow import FairSupConLoss
            self.loss = FairSupConLoss()
        elif self.loss_name == 'fscl*':
            from uow import FairSupConLoss
            self.loss = FairSupConLoss()
        elif self.loss_name == 'endunsupervised':
            from src.end import EnDUnsupervised
            self.loss = EnDUnsupervised(self.params)
        elif self.loss_name == 'lejepa':
            from src.sigreg import LEJEPA
            # LeJEPA shares the suite-wide regularization coefficient. Keep an
            # explicit legacy override so older launch scripts remain usable.
            legacy_lambda = getattr(self.params, 'lambda_lejepa', None)
            lambda_lejepa = (
                getattr(self.params, 'lambda_reg', 0.05)
                if legacy_lambda is None
                else legacy_lambda
            )
            self.loss = LEJEPA(
                lambd=lambda_lejepa,
                nb_global_views=getattr(self.params, 'nb_global_views', 2),
            )
        elif self.use_cross_entropy:
            self.loss = nn.CrossEntropyLoss()
        else:
            from uow import OW
            self.loss = OW(self.params)

        if hasattr(self.loss, "reg") and hasattr(self.loss.reg, "kernel"):
            kernel_inputs = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32, device=self.device)
            print(
                "Regularization kernel values for distances [0.0, 0.5, 1.0]: "
                f"{self.loss.reg.kernel(kernel_inputs)}" # type: ignore
            )
    
    def forward(self, xs):
        embs = [self.f(x) for x in xs] # type: ignore
        # if not self.use_cross_entropy and self.params.reg_name not in {'sigreg', 'sigreginduced'}:
        #     embs  = [F.normalize(emb, dim=-1) for emb in embs]
        zs = [self.g(emb) for emb in embs]
        if not self.use_cross_entropy and self.params.reg_name not in {'sigreg', 'sigreginduced'}:
            zs  = [F.normalize(z, dim=-1) for z in zs]
        return zs, embs
    
    def info_loss(self, batch, batch_idx, mode='train'):
        xs, bias, y = batch

        zs, embs = self.forward(xs)

        if self.loss_name == 'fscl+':
            features = torch.stack(zs, dim=1)
            L = self.loss(
                features=features, labels=y, sensitive_labels=bias, 
                group_norm=1, method="FSCL" 
            )
            L_dict = {f"{mode}/L_FSCL+": L.item()}
            
        elif self.loss_name == 'fscl*':
            features = torch.stack(zs, dim=1)
            dummy_labels = torch.zeros_like(bias)
            L = self.loss(
                features=features, labels=dummy_labels, sensitive_labels=bias, 
                group_norm=0, method="FSCL*"
            )
            L_dict = {f"{mode}/L_FSCL*": L.item()}
        elif self.use_cross_entropy:
            L = sum(self.loss(logits, y) for logits in zs) / len(zs)
            L_dict = {f"{mode}/L_CE": L.item()} #type: ignore
        elif self.loss_name == 'lejepa':
            L, L_dict = self.loss(zs, mode=mode)
        else:
            L, L_dict = self.loss(zs, bias, mode=mode)

        if self.linear_prob_labels is not None:
            embs_ = [emb.detach() for emb in embs]

            yhat_labels = [self.linear_prob_labels(emb_) for emb_ in embs_]
            prob_loss_labels = sum(F.cross_entropy(logits, y) for logits in yhat_labels) / len(yhat_labels)
            L_dict[f"{mode}/prob_l"] = (prob_loss_labels).item() # type: ignore
            total_loss = L + prob_loss_labels
        else:
            total_loss = L

        return total_loss, L_dict

    def training_step(self, batch, batch_idx, mode='train'):
        total_loss, metrics = self.info_loss(batch, batch_idx, mode=mode)
        self.log_dict(metrics, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        return total_loss

    def validation_regularizer(self, z, bias):
        """Compute a single-view regularizer for validation monitoring."""
        if self.loss_name in {'uow', 'uow_intersectional'}:
            return self.loss.reg(z, bias)
        if self.loss_name == 'ow':
            return self.loss.reg(z)
        if self.loss_name == 'lejepa':
            return self.loss.loss_reg(z) * z.size(0)
        if self.loss_name == 'endunsupervised':
            return self.loss.reg(z, bias)
        return None

    
    def validation_step(self, batch, batch_idx, mode='val'):
        x, bias, y = batch
        xs = [x]
        zs, embs = self.forward(xs)

        val_lreg = self.validation_regularizer(zs[0], bias)
        if val_lreg is not None:
            self.log(
                f"{mode}/Lreg",
                val_lreg,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=x.shape[0],
            )

        if self.use_cross_entropy:
            preds_labels = zs[0].argmax(dim=-1)
            self.val_preds_labels.append(preds_labels.detach())
            self.val_y.append(y.detach())
            self.val_bias.append(bias.detach())
            return None

        if self.linear_prob_labels is not None:
            emb = embs[0].detach()
            logits_labels = self.linear_prob_labels(emb)
            preds_labels = logits_labels.argmax(dim=-1)

            self.val_preds_labels.append(preds_labels.detach())
            self.val_y.append(y.detach())
            self.val_bias.append(bias.detach())
        return None

    def on_validation_epoch_end(self, mode='val'):
        if (self.linear_prob_labels is not None or self.use_cross_entropy) and self.val_preds_labels:
            preds_labels = torch.cat(self.val_preds_labels).to(self.device)
            y = torch.cat(self.val_y).to(self.device)
            bias = torch.cat(self.val_bias).to(self.device)

            labels_acc = (preds_labels == y).float().mean()
            wga = compute_wga_torch(y, preds_labels, bias)
            eo = compute_equalized_odds_torch(y, preds_labels, bias)

            self.log(f"{mode}/Acc", labels_acc.to(self.device), prog_bar=True, sync_dist=True)
            self.log(f"{mode}/WGA", wga.to(self.device), prog_bar=True, sync_dist=True)
            self.log(f"{mode}/EO", eo.to(self.device), prog_bar=True, sync_dist=True)

            self.val_preds_labels.clear()
            self.val_y.clear()
            self.val_bias.clear()
                    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            list(self.parameters()),
            lr=self.params.lr,
            weight_decay=self.params.weight_decay,
        )

        steps_per_epoch = self.trainer.num_training_batches
        warmup_steps = max(1, steps_per_epoch)  # 1 full epoch of warmup
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps) # type: ignore
        constant = torch.optim.lr_scheduler.ConstantLR(
            optimizer,
            factor=1.0,
            total_iters=self.trainer.max_epochs * steps_per_epoch - warmup_steps, # type: ignore
        ) # type: ignore
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, constant], milestones=[warmup_steps]) # type: ignore

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

class PretrainedWithRetrieval(pl.LightningModule):
    def __init__(self,  params):
        super().__init__()  
        self.save_hyperparameters(params.__dict__)  
        self.params = params        

        self.name_data = str(getattr(self.params, 'name_data', 'synthetic_engraving'))

        self.loss_name = getattr(self.params, 'loss_name', 'uow')

        if self.name_data == 'synthetic_engraving':
            backbone = getattr(self.params, 'backbone', 'dinov3_convnext')
            if backbone == 'vit_s8':
                self.f = DINOv1BaseModel(in_chans=1)
                if self.f.model.config.patch_size != 8:
                    raise ValueError("The pretrained ViT-S/8 must use 8x8 patches")
                in_dim = self.f.model.config.hidden_size
            elif backbone in {'resnet18', 'resnet50'}:
                import torchvision
                if backbone == 'resnet18':
                    weights = torchvision.models.ResNet18_Weights.DEFAULT
                    resnet = torchvision.models.resnet18(weights=weights)
                else:
                    weights = torchvision.models.ResNet50_Weights.DEFAULT
                    resnet = torchvision.models.resnet50(weights=weights)
                old_conv1 = resnet.conv1
                resnet.conv1 = nn.Conv2d(
                    1,
                    old_conv1.out_channels,
                    kernel_size=old_conv1.kernel_size,
                    stride=old_conv1.stride,
                    padding=old_conv1.padding,
                    bias=old_conv1.bias is not None,
                )
                with torch.no_grad():
                    resnet.conv1.weight.copy_(old_conv1.weight.mean(dim=1, keepdim=True))
                    if old_conv1.bias is not None:
                        resnet.conv1.bias.copy_(old_conv1.bias)
                in_dim = resnet.fc.in_features
                resnet.fc = nn.Identity()
                self.f = resnet
            elif backbone == 'dinov3_convnext':
                self.f = ConvNeXtDINOv3BaseModel(in_chans=1)
                in_dim = 384 * 2
            else:
                raise ValueError(f"Unknown synthetic engraving backbone: {backbone}")

            self.g = MLP(
                in_channels=in_dim,
                hidden_channels=[self.params.proj_out_dim],
                activation_layer=nn.GELU,
                norm_layer=nn.BatchNorm1d,
                dropout=0.0,
            )

        else:
            raise ValueError(f"Unknown dataset {self.name_data}")

        if self.loss_name == 'uow':
            from uow import UOW
            self.loss = UOW(self.params)
        elif self.loss_name == 'infonce':
            from uow import InfoNCEWrapper
            self.loss = InfoNCEWrapper(self.params)
        elif self.loss_name == 'endunsupervised':
            from src.end import EnDUnsupervised
            self.loss = EnDUnsupervised(self.params)
        else:
            from uow import OW
            self.loss = OW(self.params)

    def forward(self, xs):
        embs = [self.f(x) for x in xs] # type: ignore
        zs = [self.g(emb) for emb in embs]
        zs  = [F.normalize(z, dim=-1) for z in zs]
        return zs, embs
    
    def info_loss(self, batch, batch_idx, mode='train'):
        xs, bias = batch
        zs, embs = self.forward(xs)
        if self.loss_name == 'endunsupervised':
            if len(zs) != 2:
                raise ValueError(f"EnDUnsupervised expects exactly 2 views, got {len(zs)}.")
            L, L_dict = self.loss(zs[0], zs[1], bias, mode=mode)
        else:
            L, L_dict = self.loss(zs, bias, mode=mode)

        # if mode == 'val' or mode == 'test':
        #     emb1 = embs[0]
        #     emb2 = embs[1]
        #     z1   = zs[0]
        #     z2   = zs[1]
        #     sim_emb = - (emb1.unsqueeze(0) - emb2.unsqueeze(1)).pow(2).mean(dim=-1) #(B,B)
        #     sim_z   = - (  z1.unsqueeze(0) -   z2.unsqueeze(1)).pow(2).mean(dim=-1) #(B,B)
            
        #     logs_emb = self.compute_eval_stats(sim_emb, suffix="/emb", mode=mode)
        #     logs_z   = self.compute_eval_stats(sim_z  , suffix="/z"  , mode=mode)
        #     metrics = {**logs_emb, **logs_z}

        #     metrics = {**logs_emb, **logs_z}
        # else: 
        metrics = L_dict
        return L, metrics

    def training_step(self, batch, batch_idx, mode='train'):
        total_loss, metrics = self.info_loss(batch, batch_idx, mode=mode)
        self.log_dict(metrics, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        return total_loss


    def validation_step(self, batch, batch_idx, mode='val'):
        xs, bias = batch
        zs, embs = self.forward(xs)

        def retrieval_metrics(vectors, suffix):
            za, zb = vectors
            z_sici = F.normalize(za, dim=-1)
            z_sicj = F.normalize(zb, dim=-1)

            sim     = torch.einsum('bd, cd->bc', za, zb)


            metrics_i = self.compute_eval_stats(sim, mode=mode, suffix=suffix)
            metrics_j = self.compute_eval_stats(sim, mode=mode, suffix=suffix)
            metrics = {k: (metrics_i[k] + metrics_j[k]) / 2 for k in metrics_i.keys()}
            return metrics

        metrics = {
            **retrieval_metrics(embs, suffix="/emb"),
            **retrieval_metrics(zs, suffix="/z"),
        }
        self.log_dict(metrics, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, add_dataloader_idx=False)
        return None
     
    def test_step(self, batch, batch_idx, mode='test'):
        sici, sicj = batch
        xs = [sici, sicj]
        zs, embs = self.forward(xs)

        def retrieval_metrics(vectors, suffix):
            z_sici, z_sicj = vectors
            z_sici = F.normalize(z_sici, dim=-1)
            z_sicj = F.normalize(z_sicj, dim=-1)

            sim_i     = torch.einsum('bd, cd->bc', z_sici, z_sici)
            sim_j     = torch.einsum('bd, cd->bc', z_sicj, z_sicj)
            sim_cross = torch.einsum('bd, cd->bc', z_sici, z_sicj)

            diag_values = sim_cross.diag()
            off_diag_mask = ~torch.eye(sim_i.size(0), dtype=torch.bool, device=sim_i.device)
            same_mean = diag_values.mean()
            notsame_mean = (sim_i[off_diag_mask].mean() + sim_j[off_diag_mask].mean()) / 2

            sim_i[range(sim_i.size(0)), range(sim_i.size(1))] = diag_values
            sim_j[range(sim_j.size(0)), range(sim_j.size(1))] = diag_values

            metrics_i = self.compute_eval_stats(sim_i, mode=mode, suffix=suffix)
            metrics_j = self.compute_eval_stats(sim_j, mode=mode, suffix=suffix)
            metrics = {k: (metrics_i[k] + metrics_j[k]) / 2 for k in metrics_i.keys()}
            if suffix == "/z":
                metrics[f"{mode}/same{suffix}"] = same_mean
                metrics[f"{mode}/notsame{suffix}"] = notsame_mean
            return metrics

        metrics = {
            **retrieval_metrics(embs, suffix="/emb"),
            **retrieval_metrics(zs, suffix="/z"),
        }
        self.log_dict(metrics, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, add_dataloader_idx=False)
        return None
                 
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            list(self.parameters()),
            lr=self.params.lr,
            weight_decay=self.params.weight_decay,
        )
        # import schedulefree
        # optimizer = schedulefree.AdamWScheduleFree(list(self.parameters()),  lr=self.params.lr)

        steps_per_epoch = 2*100
        max_epochs = self.params.epochs
        warmup_steps = max(1, steps_per_epoch)  # 1 full epoch of warmup
        print(steps_per_epoch)
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps) # type: ignore
        constant = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0 , total_iters=max_epochs*steps_per_epoch - warmup_steps) # type: ignore
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, constant], milestones=[warmup_steps]) # type: ignore
        # return optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    # def on_train_start(self) -> None:
    #     self.optimizers().train()
        
    # def on_predict_start(self) -> None:
    #     self.optimizers().eval()

    def compute_eval_stats(self, sim, mode="val", suffix="", sim_to_compare=None):
        """Compute retrieval metrics from a similarity matrix.

        Assumptions:
            - `sim` is a (B, B) similarity matrix where higher is better.
            - For each query i, the positive is the diagonal element sim[i, i]
            (or `sim_to_compare[i]` if provided).

        Returns:
            dict containing Recall@1/3/5 and mAP (single-positive AP), keyed as
            f"{mode}/<metric>{suffix}".
        """
        # Positive score per query
        if sim_to_compare is None:
            sim_to_compare = sim.diagonal()  # (B,)
        sim_to_compare = sim_to_compare.unsqueeze(-1)  # (B, 1)

        # Rank = number of candidates strictly better than the positive
        # (0 means positive is top-1)
        rankings = (sim > sim_to_compare).sum(dim=1).float()  # (B,)

        # Recall@K: positive appears in top-K <=> rank < K
        def recall_at_k(k: int) -> torch.Tensor:
            return (rankings < float(k)).float().mean()

        r1 = recall_at_k(1)
        r3 = recall_at_k(3)
        r5 = recall_at_k(5)

        # With a single relevant item per query, AP = 1 / (rank + 1)
        # (this is equivalent to MRR, but we expose it as mAP as requested)
        mAP = (1.0 / (rankings + 1.0)).mean()

        def key(k):
            return f"{mode}/{k}{suffix}"

        log_stats = {
            # key("R@1"): r1,
            # key("R@3"): r3,
            key("R@5"): r5,
            key("mAP"): mAP,
        }
        return log_stats
    
# %%
