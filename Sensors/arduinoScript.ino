void setup() {
  // initialize the serial communication:
  Serial.begin(19200);
  pinMode(10, INPUT);
  pinMode(11, INPUT);

}

void loop() {
  while(Serial.available() == 0){}
  while(Serial.available()){
    Serial.read();
  }
  //Serial.println("Sending Data...");
  while(Serial.available() == 0){
    if((digitalRead(10) == 1)||(digitalRead(11) == 1)){
      //Serial.println('!');
    }
    else{
      //send time
        Serial.print(micros());
        Serial.print(", ");
      // send the value of the ECG:
        Serial.print(analogRead(A0));
        Serial.print(", ");
      // send the value of the GSR
        Serial.println(analogRead(A1));
    }
    //Wait for a bit to keep serial data from saturating
    delay(1);
  }
  //Serial.println("Done Sending");
  
  while(Serial.available()){
    Serial.read();
  }
}


