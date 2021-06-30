import numpy as np
Matr_1 = np.zeros((10000, 100))
for i in range(4, 9802):    # добутки (максимум 99*99=9801)
    for first_numb in range(99, 2, -1):    # діапазон перших чисел
        if int(i/first_numb) < 100:  # -обмеження для другого числа (не може бути >100)
            if i % first_numb == 0 and first_numb != i and Matr_1[i, int(i/first_numb)] != 1:
                Matr_1[i, first_numb] = 1
                #print(i,"\t",Matr_prod[i,1:j+1])
Matr_1_suma = np.sum(Matr_1, axis=1)
Matr_1_suma_final = Matr_1_suma > 1
print(f"Кількість добутків у 1 ствердженні - {np.sum(Matr_1_suma_final)}")

Matr_2 = np.zeros((200, 100))
for i in range(4, 199):     # суми (максимум 99+99=198)
    first_numb = 2                 # перше число
    while first_numb < 100 and first_numb < i-1:
        if (i-first_numb) < 100:     # обмеження для другого числа (<100)
         if Matr_1_suma_final[first_numb*(i-first_numb)] and Matr_2[i, i-first_numb] != 1:
             Matr_2[i, first_numb] = 1
         elif Matr_1_suma_final[first_numb*(i-first_numb)] != 1 and Matr_2[i, i-first_numb] != 1000:
             Matr_2[i, first_numb] = 1000           # признак
        first_numb += 1
Matr_2_suma = np.sum(Matr_2, axis=1)
Matr_2_suma_final = np.logical_and(Matr_2_suma > 0, Matr_2_suma < 1000)
print(f"Кількість елементів суми у 2 ствердженні - {np.sum(Matr_2_suma_final)}")

Matr_3 = np.zeros((10000, 200))
for i in range(4, 9802):    # добутки (максимум 99*99=9801)
     first_numb = 2                 # перше число    
     while first_numb < 100 and first_numb < i-1:
         if i % first_numb == 0 and first_numb < i-1 and i/first_numb<100 and (first_numb+int(i/first_numb))<200 and Matr_3[i, int(i/first_numb)] != 1:
             if Matr_2_suma_final[first_numb+int(i/first_numb)]:
                    Matr_3[i, first_numb] = 1                
         first_numb += 1
Matr_3_suma = np.sum(Matr_3, axis=1)
Matr_3_suma_final = Matr_3_suma == 1
print(f"Кількість добутків у 3 ствердженні- {np.sum(Matr_3_suma_final)}")

Matr_4 = np.zeros((200, 100))
for i in range(4, 199):     # суми (максимум 99+99=198)
    first_numb = 2                 # перше число
    while first_numb < 100 and first_numb < i-1:
        if (i-first_numb) < 100:     # обмеження для другого числа (<100)
         if Matr_3_suma_final[first_numb*(i-first_numb)] and Matr_2_suma_final[i] and Matr_4[i, i-first_numb]!=1:
             Matr_4[i, first_numb] = 1
        first_numb += 1
Matr_4_suma = np.sum(Matr_4, axis=1)
Matr_4_suma_final = Matr_4_suma == 1
print(f"Кількість елементів суми у 4 ствердженні- {np.sum(Matr_4_suma_final)}")

suma=np.where(Matr_4_suma_final==1)[0]
first_numb=np.where(Matr_4[suma[0],:]==1)[0]
second_numb=suma[0]-first_numb

print(f'---------- \nРезультат:\t Елементи (a,b)=({first_numb[0]},{second_numb[0]});\n\t\t\t Добуток={first_numb[0]*second_numb[0]}; \n\t\t\t Сума={suma[0]}')