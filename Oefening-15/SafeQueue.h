#include <chrono>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <condition_variable>
#include <stdexcept>

class SafeQueue {
public:
	SafeQueue() {
		firstElement = 0;
		lastElement = 0;
		size = 0;
	}

	//voeg een element toe aan het einde van de rij
	void push(float f) {
		std::unique_lock<std::shared_mutex> lock(mutex_);

		not_full.wait(lock, [this]() { return size < queueSize; });

		queue[lastElement] = f;
		lastElement = (lastElement + 1) % queueSize;
		++size;

		not_empty.notify_one();
	}

	//verwijder het eerste element uit de rij
	void pop() {
		std::unique_lock<std::shared_mutex> lock(mutex_);

		not_empty.wait(lock, [this]() { return size > 0; });

		firstElement = (firstElement + 1) % queueSize;
		--size;

		not_full.notify_one();
	}

	//geef het i-de element van de rij terug (synchronisatiefout tussen getSize() en ret = queue)
	float get(unsigned int i) {
		std::shared_lock<std::shared_mutex> lock(mutex_);

		// Wacht tot er minstens (i+1) elementen zijn
		not_empty.wait(lock, [this, i]() { return size > i; });

		unsigned int index = (firstElement + i) % queueSize;
		return queue[index];
	}


	//geef het aantal elementen in de rij terug
	unsigned int getSize() {
		std::shared_lock<std::shared_mutex> lock(mutex_);
		return size;
	}

private:
	static const unsigned int queueSize = 2;
	float queue[queueSize];

	unsigned int firstElement;
	unsigned int lastElement;
	unsigned int size;

	mutable std::shared_mutex mutex_;
	std::condition_variable_any not_empty;
	std::condition_variable_any not_full;
};
