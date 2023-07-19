import matplotlib.pyplot as plt
import seaborn as sns


def plot_students_count(data):
    sns.lineplot(
        x="Metų pabaiga",
        y="predictions",
        data=data,
        hue="BU Institucijos savivaldybė",
    )
    plt.xlabel("Mokslo Metai")
    plt.ylabel("Studentų skaičius")
    return plt
