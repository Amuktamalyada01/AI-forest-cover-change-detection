//AOI
var adilabad = table
  .filter(ee.Filter.eq('NAME_2', 'Adilabad'))
  .geometry();

Map.centerObject(adilabad, 9);
Map.addLayer(adilabad, {color: 'black'}, 'AOI');

//CLOUD MASK
function maskLandsat(image) {
  var qa = image.select('QA_PIXEL');
  
  var mask = qa.bitwiseAnd(1 << 3).eq(0)
               .and(qa.bitwiseAnd(1 << 5).eq(0));
  
  return image.updateMask(mask)
              .multiply(0.0000275)
              .add(-0.2);
}

//LANDSAT COMPOSITES
// 2000 (Landsat 5)
var ls2000 = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")
  .filterBounds(adilabad)
  .filterDate('2000-01-01', '2000-12-31')
  .map(maskLandsat)
  .median()
  .clip(adilabad);

// 2023 (Landsat 8)
var ls2023 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
  .filterBounds(adilabad)
  .filterDate('2023-01-01', '2023-12-31')
  .map(maskLandsat)
  .median()
  .clip(adilabad);

// RGB Visualization
Map.addLayer(ls2023.select(['SR_B5','SR_B4','SR_B3']),
  {min:0, max:0.3}, 'Landsat 2023 RGB');

//NDVI CALCULATION
var ndvi2000 = ls2000.normalizedDifference(['SR_B4','SR_B3'])
  .rename('NDVI_2000');

var ndvi2023 = ls2023.normalizedDifference(['SR_B5','SR_B4'])
  .rename('NDVI_2023');

Map.addLayer(ndvi2000, 
  {min:0, max:1, palette:['brown','yellow','green']},
  'NDVI 2000');

Map.addLayer(ndvi2023, 
  {min:0, max:1, palette:['brown','yellow','green']},
  'NDVI 2023');

//NDVI DIFFERENCE
var ndviChange = ndvi2023.subtract(ndvi2000)
  .rename('NDVI_Change');

Map.addLayer(ndviChange,
  {min:-0.5, max:0.5, palette:['red','white','blue']},
  'NDVI Change');


//ML TRAINING DATA
var preliminaryClass = ndviChange.expression(
  "(b < -0.2) ? 0" +
  ": (b > 0.2) ? 2" +
  ": 1", {
    'b': ndviChange
}).rename('class');

var featureStack = ndvi2000
  .addBands(ndvi2023)
  .addBands(ndviChange)
  .addBands(preliminaryClass);

var trainingSample = featureStack.sample({
  region: adilabad,
  scale: 60,
  numPixels: 1000, //safer memory
  seed: 42,
  geometries: false
});

//RANDOM FOREST MODEL
var classifier = ee.Classifier.smileRandomForest(50)
  .train({
    features: trainingSample,
    classProperty: 'class',
    inputProperties: ['NDVI_2000','NDVI_2023','NDVI_Change']
  });

var mlClassification = featureStack
  .select(['NDVI_2000','NDVI_2023','NDVI_Change'])
  .classify(classifier);

Map.addLayer(mlClassification,
  {min:0, max:2, palette:['red','yellow','green']},
  'ML Forest Change');

//ACCURACY
var validation = trainingSample.classify(classifier);
var confusionMatrix = validation.errorMatrix('class', 'classification');

print('Overall Accuracy:', confusionMatrix.accuracy());
print('Confusion Matrix:', confusionMatrix);

//AREA STATISTICS
var areaImage = ee.Image.pixelArea().divide(1e6);

var lossArea = areaImage.updateMask(mlClassification.eq(0))
  .reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: adilabad,
    scale: 120,
    bestEffort: true
  });

var stableArea = areaImage.updateMask(mlClassification.eq(1))
  .reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: adilabad,
    scale: 120,
    bestEffort: true
  });

var gainArea = areaImage.updateMask(mlClassification.eq(2))
  .reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: adilabad,
    scale: 120,
    bestEffort: true
  });

print('Forest Loss Area (sq.km):', lossArea);
print('Stable Area (sq.km):', stableArea);
print('Forest Gain Area (sq.km):', gainArea);

Export.image.toDrive({
  image: mlClassification,
  description: 'Adilabad_ML_Forest_Change_2000_2023',
  region: adilabad,
  scale: 30,
  maxPixels: 1e13
});
