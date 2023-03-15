import math
import os
import sys
import matplotlib
import torch
import torch.nn as nn

#import seaborn as sns
#sns.set_style("whitegrid")

from PIL import Image
import matplotlib.pyplot as plt
# plt.rc('font',family='Times New Roman')
import numpy as np
from timm.data.transforms import str_to_interp_mode
from torchvision import datasets, transforms



def unbiased_HSIC(K, L):
    '''Computes an unbiased estimator of HISC. This is equation (2) from the paper'''

    #create the unit **vector** filled with ones
    n = K.shape[0]
    ones = np.ones(shape=(n))

    #fill the diagonal entries with zeros
    np.fill_diagonal(K, val=0) #this is now K_tilde
    np.fill_diagonal(L, val=0) #this is now L_tilde

    #first part in the square brackets
    trace = np.trace(np.dot(K, L))

    #middle part in the square brackets
    nominator1 = np.dot(np.dot(ones.T, K), ones)
    nominator2 = np.dot(np.dot(ones.T, L), ones)
    denominator = (n-1)*(n-2)
    middle = np.dot(nominator1, nominator2) / denominator


    #third part in the square brackets
    multiplier1 = 2/(n-2)
    multiplier2 = np.dot(np.dot(ones.T, K), np.dot(L, ones))
    last = multiplier1 * multiplier2

    #complete equation
    unbiased_hsic = 1/(n*(n-3)) * (trace + middle - last)

    return unbiased_hsic

def CKA(X, Y):
    '''Computes the CKA of two matrices. This is equation (1) from the paper'''

    nominator = unbiased_HSIC(np.dot(X, X.T), np.dot(Y, Y.T))
    denominator1 = unbiased_HSIC(np.dot(X, X.T), np.dot(X, X.T))
    denominator2 = unbiased_HSIC(np.dot(Y, Y.T), np.dot(Y, Y.T))
    cka = nominator/np.sqrt(denominator1*denominator2)

    return cka

def calculate_CKA_for_two_matrices(activationA, activationB):
    '''Takes two activations A and B and computes the linear CKA to measure their similarity'''

    #unfold the activations, that is make a (n, h*w*c) representation
    shape = activationA.shape
    activationA = np.reshape(activationA, newshape=(shape[0], np.prod(shape[1:])))

    shape = activationB.shape
    activationB = np.reshape(activationB, newshape=(shape[0], np.prod(shape[1:])))

    #calculate the CKA score
    cka_score = CKA(activationA, activationB)

    del activationA
    del activationB

    return cka_score


@ torch.no_grad()
def forward_hook(module, input, output):
    x = input[0]

    feat_list = [[], [], [], []]

    x = module.stem(x)
    c0, c1, c2, c3 = 0, 0, 0, 0
    for i in range(module.num_subnet):
        c0, c1, c2, c3 = getattr(module, f'subnet{str(i)}')(x, c0, c1, c2, c3)
        feat_list[0].append(c0.flatten(2).transpose(1, 2).cpu()) #.reshape(c0.size(0), -1).cpu())
        feat_list[1].append(c1.flatten(2).transpose(1, 2).cpu()) #.reshape(c1.size(0), -1).cpu())
        feat_list[2].append(c2.flatten(2).transpose(1, 2).cpu()) #.reshape(c2.size(0), -1).cpu())
        feat_list[3].append(c3.flatten(2).transpose(1, 2).cpu()) #.reshape(c3.size(0), -1).cpu())
    #logits = module.cls_blocks[-1](c3).softmax(dim=-1)
    return feat_list
    #return x.flatten(2).transpose(1, 2).reshape(x.size(0), -1).cpu(), feat_list, logits.cpu()


def build_data_loader(batch_size):
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=str_to_interp_mode("bicubic")),  # to maintain same ratio w.r.t. 224 images
        transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    dataset = datasets.ImageFolder(root, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    return data_loader


def load_backbone_model(model, ckpt_path):
    ckpt = torch.load(ckpt_path)
    if 'model' in ckpt.keys():
        ckpt = ckpt['model']
    new_ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    print(model.load_state_dict(new_ckpt, strict=False))
    model.eval()
    return model


def plot_fig(image, feat_list, save_dir_path, fig_name, score):
    N = len(feat_list)
    assert N%2 == 0
    fig, subs = plt.subplots(2, 5, figsize=(10, 5))
    subs = [subs[i][j] for i in range(2) for j in range(5)]
    from sklearn.decomposition import PCA
    pca = PCA(n_components=50)

    subs[0].imshow(image)
    subs[0].set_axis_off()
    subs[0].set_title("{:.2f}".format(score.cpu().numpy().round(2)))
    for i in range(N):
        tmp = pca.fit_transform(feat_list[i])
        tmp = torch.tensor(tmp).reshape(7, 7, -1)
        subs[i+1].imshow(tmp)
        subs[i+1].set_axis_off()
    plt.savefig(os.path.join(save_dir_path, fig_name+'.png'))


@torch.no_grad()
def visualize(model, save_dir_path):
    data_loader = build_data_loader(1)

    for i, (data, target) in enumerate(data_loader):
        data = data.cuda()
        target = target.cuda()
        import pdb; pdb.set_trace()
        logits, feat_list  = model(data)
        score = logits.max()
        acc = (logits.argmax() == target)
        feat_list = [feat.cpu() for feat in feat_list[-1]]
        data = data[0].permute(1, 2, 0).cpu()
        data = data * torch.tensor([0.229, 0.224, 0.225]).view(1, 1, -1) + torch.tensor([0.229, 0.224, 0.225]).view(1, 1, -1)
        if score > 0.7 and acc:
            plot_fig(data, feat_list, save_dir_path, './images/%05d'%i, score)
        if i > 500:
            break

def cal_cka_per_list(feat_list, target):
    res = []
    for feat in feat_list:
        res.append(calculate_CKA_for_two_matrices(feat.float().numpy(), target.float().numpy()))
    return res

@torch.no_grad()
def cal_correlation(feat_list):
    res = []
    for feat in feat_list:
        # print(feat.shape)
        feat = feat.flatten(2)
        corr = torch.softmax(torch.bmm(feat.transpose(1, 2), feat).div(0.07), dim=-1).mean(dim=0)
        # print(corr.shape)
        res.append(corr.cpu().numpy())
    return res


def plot_correlation(feat_list, save_dir_path):
    fig, subs = plt.subplots(1, 4, figsize=(10, 2), sharey=False)

    for i in range(4):
        corr_res = cal_correlation(feat_list[i])
        subs[i].imshow(corr_res[-1])

    plt.savefig(os.path.join(save_dir_path, 'plot_correlation.png'), bbox_inches='tight')


def plot_cka(images, targets, feat_list, save_dir_path):

    one_hots = torch.nn.functional.one_hot(targets, num_classes=1000) + 1e-6
    images = images.permute(0, 2, 3, 1).flatten(1)
    cka_ihx = []
    cka_ihy = []
    print("Calculating CKA between features and input/output...")
    for i in range(4):
        cka_image = cal_cka_per_list(feat_list[i], images)
        cka_target = cal_cka_per_list(feat_list[i], one_hots)
        cka_ihx.append(cka_image)
        cka_ihy.append(cka_target)
        # subs[i].plot(range(len(cka_image)), cka_image, label=legend[0])
        # subs[i].plot(range(len(cka_image)), cka_target, label=legend[1])

        # subs[i].set_title(titles[i])
        # subs[i].set_ylim(0, 1.0)
        # subs[i].set_xlabel('Column')
    # legend = ["image and feature", "image and label"]
    print("Calculating Done! Visualizing")
    fig, subs = plt.subplots(1, 2, figsize=(10, 3))
    titles = ["CKA similarity between features and images", "CKA similarity between features and labels"]
    # cka_ihx=cka_ihx/np.linalg.norm(np.array(cka_ihx))
    # cka_ihy=cka_ihy/np.linalg.norm(np.array(cka_ihy))
    # subs[0].imshow(cka_ihx)
    # subs[0].set_title(titles[0])
    # subs[1].imshow(cka_ihy)
    # subs[1].set_title(titles[1])
    # subs[0].set_ylabel('CKA Similarity')

    # # plt.legend(bbox_to_anchor=(-2.5, -0.45, 2, 1), loc=4, ncol=2, mode='expand', borderaxespad=0.)
    # plt.xticks([])
    # plt.yticks([])
    # plt.axis('off')
    row_name = ["col"+str(i) for i in range(len(feat_list[0]))]
    im, cbar = heatmap(np.array(cka_ihx), col_labels=row_name, row_labels=['c0', 'c1', 'c2', 'c3'], ax=subs[0], cbarlabel=titles[0])
    texts = annotate_heatmap(im, valfmt="{x:.2f}")
    im, cbar = heatmap(np.array(cka_ihy), col_labels=row_name, row_labels=['c0', 'c1', 'c2', 'c3'], ax=subs[1], cbarlabel=titles[1])
    texts = annotate_heatmap(im, valfmt="{x:.2f}")

    plt.savefig(os.path.join(save_dir_path, 'cka.png'), dpi=220)


@torch.no_grad()
def cka(model, save_dir_path):
    data_loader = build_data_loader(512)
    for i, (data, target) in enumerate(data_loader):
        print("Inference Image...")
        feat_list = model(data.cuda())
        plot_cka(data, target, feat_list, save_dir_path)
        return


@torch.no_grad()
def correlation(model, save_dir_path):
    data_loader = build_data_loader(128)
    for i, (data, target) in enumerate(data_loader):
        feat_list = model(data.cuda())
        plot_correlation(feat_list, save_dir_path, model.num_subnet)
        return


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def main(argv):
    assert argv[1] and argv[2] is not None
    ckpt_path = argv[1]
    save_dir_path = argv[2]
    os.makedirs(save_dir_path, exist_ok=True)
    from models.revcol import revcol_small, revcol_tiny
    model = revcol_small(save_memory=False, inter_supv=True)
    model.register_forward_hook(forward_hook)
    model = load_backbone_model(model, ckpt_path)
    model = model.cuda()

    # visualize(model, save_dir_path)
    cka(model, save_dir_path)
    # correlation(model, save_dir_path)
    print("Finished. Please check image folder {} for the results.".format(save_dir_path))


if __name__ == '__main__':
    main(sys.argv)
