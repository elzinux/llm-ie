from src.eval.semantic_eval.scorer import score_semantic_eval

arr = [""] * 8
arr[0] = (
    "Amoxicillin is a type of antibiotic. It is mainly used to treat a variety of bacterial infections. For example, it can be used to treat infections of the middle ear, sinuses, throat, lungs, urinary tract and skin. It works by stopping the growth of bacteria. However, it is important to note that it is only effective against bacterial infections and not viral infections."  # 0.91
)
arr[1] = (
    "Amoxicillin is a penicillin - like antibiotic. It works by stopping the growth of bacteria. It is used to treat a variety of bacterial infections, such as infections of the middle ear, tonsils, throat, urinary tract, skin, and respiratory tract (like pneumonia, bronchitis). However, it is ineffective against viral infections."
)
arr[2] = (
    "Shanghai captivates me with its seamless blend of history and modernity. Strolling along the Bund at dusk, watching the neon-lit skyline of Pudong reflect on the Huangpu River, fills me with awe. The city’s energy is palpable—a dynamic rhythm that pulses through its bustling streets, art deco buildings, and hidden lanes filled with local snacks like xiaolongbao. Whether exploring the timeless charm of Yuyuan Garden or marveling at the futuristic towers of Lujiazui, Shanghai never fails to inspire. It’s a place where tradition and innovation coexist, creating a magnetic allure that feels like home."  # 0.75
)
arr[3] = (
    "I’m deeply enamored with Shanghai’s ability to bridge past and present. The Bund’s colonial architecture stands in striking contrast to the glowing skyscrapers of Pudong, a visual symphony of eras. There’s an indescribable vitality here—a buzz in the air from the crowds, the clatter of woks in street-side kitchens, and the hum of ambition in its tech-driven districts. From sipping tea in a century-old teahouse to losing myself in the neon-drenched chaos of Nanjing Road, Shanghai offers a mosaic of experiences. It’s a city that embraces change while honoring its roots, and this duality is what makes my heart swell with admiration every time I wander its streets."
)
arr[4] = (
    "Living in the city is an exhilarating adventure filled with endless possibilities. The vibrant energy of bustling streets, diverse cultures, and innovative opportunities invigorate the spirit. Every corner holds a new experience—a cozy café tucked between skyscrapers, a hidden gallery showcasing local art, or a lively festival uniting strangers. Public transport connects you to the world, while careers in cutting-edge fields thrive here. Though chaos sometimes lingers, it’s the chaos of progress, of humanity collaborating and dreaming aloud. The city is a living, breathing testament to what we can achieve when we embrace curiosity and connection."  # 0.51
)
arr[5] = (
    "Urban existence feels like a relentless marathon, where peace is a luxury few can afford. The constant roar of traffic, the suffocating crowds, and the pressure to 'keep up’ drain the soul. Skyscrapers cast long shadows over parks once meant for tranquility, replacing greenery with concrete. Relationships grow superficial, forged in hurried elevator rides or digital screens rather than genuine dialogue. Opportunities? They exist, but at the cost of sanity—long commutes, skyrocketing costs, and a sense of being just another cog in a machine. The city promises progress but delivers isolation, a cacophony of noise drowning out the quiet moments that make life meaningful."
)
arr[6] = "I like you."
arr[7] = "I hate you."

# score = score_semantic_eval(arr[6], arr[7])
# print(score)

res = []
for i in range(0, 8, 2):
    x = arr[i]
    y = arr[i + 1]
    score = score_semantic_eval(x, y)
    res.append(score)
print(res)
# 原始：[0.91, 0.75, 0.51, 0.53]
# [0.94, 0.588, 0.4358, 0.6604]
# [0.44795772433280945, 0.27629607915878296, 0.22147083282470703, 1.0]
# [0.96, 0.63, 0.56, 0.66]
# [0.9265, 0.5862, 0.465, 0.6604]
