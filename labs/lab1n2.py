import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import numpy as np
import os


def lab1(n_var=1, flag=0):
    # Создаем путь к директории "data" в текущей директории
    global data_dir, output_dir, fig, axs
    data_dir = os.path.normpath(os.path.join(os.getcwd(), "files_for_lab", "lab1n2", "input"))
    output_dir = os.path.normpath(os.path.join(os.getcwd(), "files_for_lab", "lab1n2", "output"))
    matplotlib.font_manager.fontManager.addfont(os.path.normpath(os.path.join(os.getcwd(), data_dir, 'Courier.ttf')))

    # Создаем путь к файлу "example.txt" в директории "data"
    pict_path = os.path.normpath(os.path.join(data_dir,  "my.jpg"))

    # создаем многооконный рисунок
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(16, 12))

    # задание 1
    # загружаем и отображаем свою фотографию
    with cbook.get_sample_data(pict_path) as image_file:
        img = plt.imread(image_file)
    axs[0, 0].imshow(img)
    axs[0, 0].axis('off')
    axs[0, 0].text(360, 50, u"Сложно", fontdict={'family': 'Comic Sans MS', 
                    'size': 15, 
                    'weight': 'normal', 
                    'style': 'oblique', 
                    'color': (0.9, 0.1, 0.5, 0.9)})


    # создаем стандартный график с заливкой

    #Создаем фходной файл-----------------------------------------------------
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    y2 = np.log(x)
    np.savetxt(f"{data_dir}/data.txt", [x], fmt='%f')
    with open(f"{data_dir}/data.txt", 'a') as f:
        np.savetxt(f, [y], fmt='%f')
        np.savetxt(f, [y2], fmt='%f')
    #------------------------------------------------------------------------
    data = []
    with open(os.path.normpath(os.path.join(data_dir, "data.txt"))) as f:
        for line in f:
            data.append([float(x) for x in line.split()])

    axs[0, 1].fill_between(data[0], data[n_var], where=(np.array(data[n_var]) > 0), color='lightgreen')
    axs[0, 1].fill_between(data[0], data[n_var], where=(np.array(data[n_var]) < 0), color='magenta')
    axs[0, 1].plot(data[0], data[n_var], color='blue')
    # axs[0, 1].fill_between(data[0], data[n_var + 1], where=(np.array(data[n_var+1]) > 0), color='lightgreen')
    # axs[0, 1].fill_between(data[0], data[n_var + 1], where=(np.array(data[n_var+1]) < 0), color='magenta')
    # axs[0, 1].plot(data[0], data[n_var + 1], color='blue')
    axs[0, 1].grid(True)
    # подписи 
    axs[0, 1].set_xlabel('x', fontsize=12, fontweight='bold', fontname='Courier', rotation=0)
    axs[0, 1].set_ylabel('y', fontsize=12, fontweight='bold', fontname='Courier', rotation=90)
    axs[0, 1].tick_params(axis='both', which='major', labelsize=10, rotation=60)
    if flag == 0:    
        output_pict_path = os.path.normpath(os.path.join(output_dir,  "lab1.png"))

        plt.savefig(output_pict_path)
        plt.show()

# задание 2
def lab2(n_var):
    lab1(n_var, flag=1)
    # Задаем параметры шрифта
    font = {
        'family': 'Times New Roman',
        'size': 12,
        'weight': 'bold',
        'style': 'oblique'
    }
    # Задаем цвета
    text_color = (0.4, 0.1, 0.3, 0.9)
    fill_color = (0.4, 0.1, 0.1, 0.5)
    arrow_color = '#FF0000'  # Красный цвет в формате HEX
    
    #Generation of data_sample----------------------------------------------
    import random

    # Открываем файл на запись
    with open(os.path.normpath(os.path.join(data_dir,  "fig8.txt")), "w") as file:
        # Генерируем 10 строк
        for i in range(1, 11):
            # Нечетная строка
            if i % 2 == 1:
                start = random.randint(1, 100)
                end = random.randint(start, 100)
                file.write(f"{start} {end}\n")
            # Четная строка
            else:
                data = list(range(start, end+1))
                file.write(" ".join(str(random.randint(start, end)) for _ in range(end - start + 1)) + "\n")
    #------------------------------------------------------------------------------
    # создаем столбчатую диаграмму с аннотацией
    n_line = 2 * n_var - 1
    data = []
    with open(os.path.normpath(os.path.join(data_dir, "fig8.txt")), "r") as bar_file:
        for num,line in enumerate(bar_file):
            if (num + 1) == n_line:
                ll = line.split()
                tmp = []
                for el in ll:
                    tmp.append(int(el))
                data.append(tmp)
                tmp = []
                ll = next(bar_file).split()
                for el in ll:
                    tmp.append(int(el))
                data.append(tmp)
           
    x = [i for i in range(data[0][0],data[0][1]+1)]
    y = data[1]
    axs[0, 2].bar(x, y, color=fill_color)
    # Задаем цвет текста
    plt.setp(axs[0, 2].get_xticklabels(), color=text_color)
    plt.setp(axs[0, 2].get_yticklabels(), color=text_color)

    # Добавляем аннотацию с указателем
    selected_x = np.random.choice(x)
    selected_y = y[selected_x - min(x)]
    arrowprops = {
        'arrowstyle': '->',
        'color': arrow_color
    }
    axs[0, 2].annotate(u'Аннотация', xy=(selected_x, selected_y), xytext=(selected_x - 1, selected_y + len(y)/3),
                arrowprops=arrowprops, fontproperties=font)

    # Задаем цвет фона диаграммы
    axs[0, 2].set_facecolor(fill_color)
    
    #Задание 2.2

    def y1(x):
        return np.sin(2*x)**2 - np.cos(6*x)**2
    
    roots = [-1.63, -0.44, 0.35, 1.9]
    p = np.poly1d(roots, r=True)

    def y2(x):
        return 0.3 * p(x)
    
    x = np.linspace(-2, 2, 100)
    y1_values = y1(x)
    y2_values = y2(x)

    axs[1, 0].plot(x, y1_values, linestyle='dashdot', label='(sin(2x))^2 - (cos(6x))^2')
    axs[1, 0].plot(x, y2_values, linestyle='dotted', label='0.3*p(x)')
    axs[1, 0].legend(loc='best', prop={'family': 'Tahoma', 'size': 13, 'weight': 'bold', 'style': 'italic'})
    plt.xticks(rotation=15)
 



    

    # задание 2.3
    N = 300
    x = np.random.rand(N)
    mu = 7
    sigma = 3
    y_1 = np.random.normal(mu, sigma, N)
    y_2 = np.random.uniform(0, 10, N)
    shape = 0.8
    scale = 1.7
    y_3 = np.random.gamma(shape, scale, N)
    axs[1, 1].scatter(x-1, y_1, c=(0.9, 1, 0), s=2, label='Нормальное распределение')
    axs[1, 1].scatter(x, y_2, c='#00FF00', s=2, label='Равномерное распределение')
    axs[1, 1].scatter(x+1, y_3, c='white', s=2, label='Гамма-распределение')
    axs[1, 1].legend(loc='best')
    axs[1, 1].set_facecolor('black')



    # задание 2.4
    # задаем параметры кривой Лиссажу
    A = 4
    B = 13
    a = 3
    b = 4
    delta = np.pi/2

    # создаем массивы для x и y значений
    t = np.linspace(0, 2*np.pi, 1000)
    x = A * np.sin(a * t + delta)
    y = B * np.sin(b * t)

    # задаем цвет заливки
    fill_color = (0.1, 0.5, 0.2, 0.5) # пример задания цвета (R,G,B,A)

    # строим график
    axs[1, 2].fill_betweenx(y, x, 0, where=x>0, interpolate=True, color=fill_color)
    axs[1, 2].fill_betweenx(y, x, 0, where=x<0, interpolate=True, color=fill_color)
    axs[1, 2].plot(x, y, color='black')

    # задаем заголовок и подписи осей
    axs[1, 2].set_title('Фигура Лиссажу', fontsize=16)
    axs[1, 2].set_xlabel('X', fontsize=12)
    axs[1, 2].set_ylabel('Y', fontsize=12)

    # задаем цвет сетки и ее прозрачность
    axs[1, 2].grid(color='red', alpha=0.5)

    # # сохраняем многооконный рисунок в файл
    output_pict_path = os.path.normpath(os.path.join(output_dir,  "lab2.png"))

    plt.savefig(output_pict_path)
    plt.show()

if __name__ == "__main__":
    n_var = 1
    lab1(n_var)
    lab2(n_var)

