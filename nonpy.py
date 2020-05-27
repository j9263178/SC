article = '''They make a pool of silver, swim as stones through the pine 
branches, push back needles like a dog's fur shaking off the 
bathwater and carry what is left of the sky as bones hitched 
together into a stream. It never tires. The stars so still are 
never really still. And what pieces you can hold-thread, whiskey, 
chigger bites, sin-you lose like the loosest water round and 
round your fingers, like her hair you cannot touch now, like 
the last bits of light you mapped on the bed until worn 
blind-through, sleepless, breath-rattled, loping through 
the dark alone.'''

a = article.split(" ")
print(a)
newa = []
for letter in a:
    if letter.__contains__("'"):
        hi = letter.split("'")
        newa.append(hi[0])
        newa.append("'")
        newa.append(hi[1])
    elif letter.__contains__("-"):
        hi = letter.split("-")
        newa.append(hi[0])
        newa.append("-")
        newa.append(hi[1])
    else:
        newa.append(letter)
nna = ""
for i in range(0, newa.__len__()-1):
    letter = newa[i]
    if letter.__contains__(",") or letter.__contains__("."):
        nna += (letter.replace(letter[-2], letter[-2].upper()))
    else:
        nna += (letter.replace(letter[-1], letter[-1].upper()))

    if letter[-1] == "." or letter[-1] == ",":
        nna += " "
    elif letter.isalpha():
        if newa[i + 1].isalpha():
            nna += " "

print(nna)
