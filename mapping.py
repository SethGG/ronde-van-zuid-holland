x = [5, 6, 9, 3]
y = [3, 5, 4, 1]

z = {}
for i in range(len(x)):
    z[x[i]] = y[i]

print(z)

k = {}
for i in z:
    if i not in z.values():
        j = z[i]
        while j in z:
            j = z[j]
        k[i] = j
        k[j] = i

print(k)
