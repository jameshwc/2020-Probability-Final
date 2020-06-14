package main

import (
	"bufio"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strings"
	"time"

	mapset "github.com/deckarep/golang-set"
)

const (
	filename   = "t1-data.csv"
	batchNum   = 10
	batchSize  = 10
	initScore  = .2
	sampleSize = 20000
	dataSize   = 2000000
)

var csvData [][]string

func hellingerDistance(normP []float64, normQ []float64) float64 {
	s := 0.0
	for i := range normP {
		s += math.Sqrt(normP[i] * normQ[i])
	}
	return math.Sqrt(1 - s)
}
func parse() {
	file, err := os.Open(filename)
	if err != nil {
		log.Fatalf("read %s", filename)
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	scanner.Split(bufio.ScanLines)
	for i := 0; scanner.Scan(); i++ {
		dat := strings.Split(scanner.Text(), ",")
		csvData = append(csvData, dat)
	}
}
func getCD(isFull bool, index mapset.Set) []map[string]int {
	dat := make([]map[string]int, len(csvData[0])-1)
	for i := range dat {
		dat[i] = make(map[string]int)
	}
	updateDat := func(row []string) {
		for idx, r := range row {
			dat[idx][r]++
		}
	}
	if isFull {
		for i := range csvData {
			updateDat(csvData[i][1:])
		}
	} else {
		for index.Cardinality() != 0 {
			updateDat(csvData[index.Pop().(int)][1:])
		}
	}
	return dat
}

func compare(fullCD []map[string]int, sampleCD []map[string]int) float64 {
	var distList []float64
	for id := range fullCD {
		var normP, normQ []float64
		pSum, qSum := 0, 0
		for k := range fullCD[id] {
			pSum += fullCD[id][k]
			qSum += sampleCD[id][k]
		}
		for k := range fullCD[id] {
			normP = append(normP, float64(fullCD[id][k])/float64(pSum))
			normQ = append(normQ, float64(sampleCD[id][k])/float64(qSum))
		}
		distList = append(distList, hellingerDistance(normP, normQ))
	}
	s := 0.0
	for i := range distList {
		s += distList[i]
	}
	return s / float64(len(distList))
}

func greedy(dataSet mapset.Set) {
	fullCD := getCD(true, nil)
	sampleSet := mapset.NewThreadUnsafeSet()
	bestScore := initScore
	batchSizeSlice := []int{100, 100, 100, 100, 1000, // 5000
		500, 500, 500, 500, 500, 500, // 8000
		300, 300, 300, 300, 300, 300, 300, // 10100
		100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
		100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
		100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
		100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
		100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
		100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
		100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
		100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
		100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
		100, 100, 100, 90, //19500
		10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
		10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
		10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
		10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
		10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
	}
	batchNumSlice := make(map[int]int)
	batchNumSlice[1000] = 100
	batchNumSlice[500] = 500
	batchNumSlice[300] = 500
	batchNumSlice[100] = 300
	batchNumSlice[90] = 400
	batchNumSlice[10] = 200

	idx := 0
	for sampleSet.Cardinality() < sampleSize {
		bestBatch := mapset.NewThreadUnsafeSet()
		curBestScore := bestScore
		for i := 0; i < batchNumSlice[batchSizeSlice[idx]]; i++ {
			dataSetCopy := dataSet.Clone()
			batch := mapset.NewThreadUnsafeSet()
			for j := 0; j < batchSizeSlice[idx]; j++ {
				batch.Add(dataSetCopy.Pop())
			}
			newSampleSet := sampleSet.Union(batch)
			sampleCD := getCD(false, newSampleSet)
			if curScore := compare(fullCD, sampleCD); curScore < curBestScore {
				bestBatch = batch
				curBestScore = curScore
			}
		}
		if bestBatch.Cardinality() != 0 {
			sampleSet = sampleSet.Union(bestBatch)
			dataSet = dataSet.Difference(bestBatch)
			bestScore = curBestScore
			fmt.Println(bestScore)
			idx++
		}
		fmt.Println(sampleSet.Cardinality())
	}
	fmt.Print(compare(fullCD, getCD(false, sampleSet)))
	fmt.Println(sampleSet)
}
func greedySet(dataSet mapset.Set, num int, batchnum int, ret chan mapset.Set, sem chan int) {
	fullCD := getCD(true, nil)
	sampleSet := mapset.NewThreadUnsafeSet()
	bestScore := initScore
	for sampleSet.Cardinality() < num {
		bestBatch := mapset.NewThreadUnsafeSet()
		curBestScore := bestScore
		for i := 0; i < 5; i++ {
			dataSetCopy := dataSet.Clone()
			batch := mapset.NewThreadUnsafeSet()
			for j := 0; j < batchnum; j++ {
				batch.Add(dataSetCopy.Pop())
			}
			newSampleSet := sampleSet.Union(batch)
			sampleCD := getCD(false, newSampleSet)
			if curScore := compare(fullCD, sampleCD); curScore < curBestScore {
				bestBatch = batch
				curBestScore = curScore
			}
		}
		if bestBatch.Cardinality() != 0 && bestScore/curBestScore >= 0.01 {
			sampleSet = sampleSet.Union(bestBatch)
			dataSet = dataSet.Difference(bestBatch)
			bestScore = curBestScore
		}
	}
	fmt.Println("in function: ", sampleSet)
	ret <- sampleSet
	<-sem
}
func goGreedy(dataSet mapset.Set) {
	sem := make(chan int, 20)
	dataCopySet := dataSet.Clone()
	sampleSet := mapset.NewThreadUnsafeSet()
	samples := make(chan mapset.Set, 2000)
	for i := 0; i < 200; i++ {
		kData := mapset.NewThreadUnsafeSet()
		fmt.Println(i)
		sem <- 1
		for j := 0; j < 10000; j++ {
			kData.Add(dataCopySet.Pop())
		}
		go greedySet(kData, 100, 10, samples, sem)
		if i%10 == 0 && i != 0 {
			for j := 0; j < 10; j++ {
				s := <-samples
				fmt.Println(s.Cardinality(), s)
				sampleSet = sampleSet.Union(s)
				fmt.Println("sampleSet: ", sampleSet.Cardinality(), sampleSet)
			}
			fmt.Println(sampleSet, compare(getCD(true, nil), getCD(false, sampleSet)))
		}
	}
	close(samples)
	for range samples {
		sampleSet = sampleSet.Union(<-samples)
	}
	fmt.Print(compare(getCD(true, nil), getCD(false, sampleSet)))
}
func greedyRemoveTopK(dataSet mapset.Set) {
	fullCD := getCD(true, nil)
	for dataSet.Cardinality() > sampleSize {
		curWorstScore := 0.0
		worstBatch := mapset.NewThreadUnsafeSet()
		for i := 0; i < batchNum; i++ {
			dataSetCopy := dataSet.Clone()
			batch := mapset.NewThreadUnsafeSet()
			for j := 0; j < 1000; j++ {
				batch.Add(dataSetCopy.Pop())
			}
			sampleCD := getCD(false, batch.Clone())
			if curScore := compare(fullCD, sampleCD); curScore > curWorstScore {
				worstBatch = batch
				curWorstScore = curScore
			}
		}
		dataSet = dataSet.Difference(worstBatch)
		fmt.Println(compare(fullCD, getCD(false, dataSet.Clone())))
	}
}
func main() {
	rand.Seed(time.Now().Unix())
	parse()
	dataSet := mapset.NewThreadUnsafeSet()
	for i := 0; i < dataSize; i++ {
		dataSet.Add(i)
	}
	// greedy(dataSet)
	goGreedy(dataSet)
	// greedyRemoveTopK(dataSet)
}
