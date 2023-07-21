#include <iostream>
#include "NeuralNetwork.h"
#include "CsvReader.h"
#include "SFML/Graphics.hpp"

using namespace std;

#define WINDOW_WIDTH	720
#define WINDOW_HEIGTH	480

int main()
{
	sf::RenderWindow window(sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGTH), "window");

	NeuralNetwork network({ 2, 5, 5, 1 });
	size_t sampleSize = 4;
	/*float sample[]{
		1, 1,
		1, 2,
		2, 1,
		2, 2
	};
	float answers[]{
		1,
		0,
		0,
		1
	};*/
	std::vector<sf::Vector2f> coords;
	std::vector<float> answers;
	std::vector<sf::CircleShape> points;
	/*std::generate(points.begin(), points.end(), [sample, answers]() {
		static size_t i = 0;
		sf::CircleShape r(5.f);
		r.setOrigin(5.f, 5.f);
		r.setPosition(sample[i*2] * 100, sample[i*2 + 1] * 100);
		r.setFillColor(answers[i] == 1.f ? sf::Color::Blue : sf::Color::Green);
		i += 1;
		return r;
		});*/
	/*CsvReader reader("Iris.csv");
	size_t sampleSize = 150;
	float* sample = new float[sampleSize * 4];
	float* answers = new float[sampleSize];
	reader.next();
	for (int i = 0; i < sampleSize; i++) {
		reader.next();
		for (int j = 0; j < 4; j++) {
			sample[i * 4 + j] = stof(reader.get(j + 1));
		}

		std::string type = reader.get(5);
		if (type.compare("Iris-setosa") == 0) {
			answers[i] = 0;
		}
		else if (type.compare("Iris-versicolor") == 0) {
			answers[i] = 1;
		}
		else if (type.compare("Iris-virginica") == 0) {
			answers[i] = 2;
		}
	}*/
	network.setLogger(nullptr)
		.setInputSample((float*)coords.data(), sampleSize)
		.setBatchSize(1)
		.setEpochs(10)
		.setLearningRate(.7)
		.setAnswers(answers.data());
		//.setTargetBatchLoss(0.001f)
		/*.train()*/;

	sf::Texture texture;
	texture.create(WINDOW_WIDTH, WINDOW_HEIGTH);

	sf::Uint8* pixels = new sf::Uint8[WINDOW_WIDTH * WINDOW_HEIGTH * 4];

	for (size_t i = 0; i < WINDOW_WIDTH * WINDOW_HEIGTH; i++)
	{
		pixels[i * 4] = 0; 
		pixels[i * 4 + 1] = 0;
		pixels[i * 4 + 2] = 0;
		pixels[i * 4 + 3] = 255;
	}
	texture.update(pixels);
	sf::Sprite sp;
	sp.setTexture(texture);

	while (window.isOpen()) {
		sf::Event event;
		while (window.pollEvent(event)) {
			if (event.type == sf::Event::Closed) {
				window.close();
			}
			if (event.type == sf::Event::MouseButtonPressed) {
				if (event.key.code == sf::Mouse::Left ||
					event.key.code == sf::Mouse::Right) {
					sf::CircleShape r(5.f);
					r.setOrigin(5.f, 5.f);
					sf::Vector2i pos = sf::Mouse::getPosition(window);
					r.setPosition(pos.x, pos.y);
					r.setFillColor(event.key.code == sf::Mouse::Left ? sf::Color::Blue : sf::Color::Green);
					points.push_back(r);
					answers.push_back(event.key.code == sf::Mouse::Left ? 1 : 0);
					coords.push_back(sf::Vector2f(pos.x / 100, pos.y / 100));
					network
						.setInputSample((float*)coords.data(), coords.size())
						.setAnswers(answers.data());
				}
			}
		}

		if (!points.empty()) {
			network.train();
			for (size_t i = 0; i < WINDOW_WIDTH * WINDOW_HEIGTH; i++)
			{
				sf::Vector2f point((float)(i % WINDOW_WIDTH) / 100, (float)i / WINDOW_WIDTH / 100);
				const float* result = network.predict((float*)&point);
				pixels[i * 4 + 1] = 200 * (1 - result[0]);
				pixels[i * 4 + 2] = result[0] * 200;
			}
			texture.update(pixels);
		}

		window.clear(sf::Color::White);
		window.draw(sp);
		std::for_each(points.begin(), points.end(), [&](auto p) { window.draw(p); });
		window.display();
	}
	return 0;
}
