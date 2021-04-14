from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def standardized_pca(x: dict, n_components: int):
    # Normalize data
    x_standardized = normalize(x)

    # PCA Reduction
    pca = PCA(n_components=n_components, copy=True)
    x_pca = pca.fit_transform(x_standardized)

    return pca, x_pca


def standardized_tsne(x: dict, n_components: int, ramdom_state: int):
    # Normalize data
    x_standardized = normalize(x)

    # tSNE Reduction
    tsne = TSNE(n_components=n_components,
                n_iter=2000, random_state=ramdom_state)
    x_tsne = tsne.fit_transform(x_standardized)

    return tsne, x_tsne


def normalize(x: dict):
    scaler = StandardScaler()
    scaler.fit(x)

    x_standardized = scaler.transform(x)

    return x_standardized
