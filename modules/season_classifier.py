import numpy as np

def classify_season(skin_lab, palettes):
    if skin_lab is None:
        return "피부 색 추출 실패"

    distances = {}
    for season, df in palettes.items():
        pal_lab = df[["L*", "a*", "b*"]].values
        dist = np.mean(np.linalg.norm(pal_lab - skin_lab, axis=1))
        distances[season] = dist

    return min(distances, key=distances.get)
